# -*- coding: utf-8 -*-
"""Phase 3139 (Omega-P137): identity
interaction decomposition + port
consumption test + L26-32 rewrite-window
causality + cross-material generalization
sentinel.

Preregistered in 3138 closeout (task 82,
aligned with FINGERPRINT_PARADIGM_PLAN.md
Omega-P137):
(1) interaction split: regress the
template directions out of the identity
component -> ish2 gate (target 0.85,
hard floor 0.75); R row-identity
retrieval (does the interaction residual
carry row identity?);
(2) port consumption: frozen 3135 dvecs
(L17/29/33/38, sha-anchored) projected
onto the Gram-Schmidt-orthogonalized
B / C / I subspaces per row -> energy
shares (prediction: I dominant, C~0);
own-row identity share recorded;
(3) rewrite-window causality: C(T2)
injection vs displaced-identity injection
(roll-1 I) at L17/26/29/32, allstep,
behavior chg on SCAN_N rows (link to
3121-3127 write chain + 3136/3137 dose
anchors);
(4) cross-material sentinel: 84 unseen
(s,o) pairs x 2 dirs captured at
L{17,29,38} -> same-subject retrieval
against the 672-row identity bank +
cross-material P/A1 AUC (Tier-1
Level-2 sentinel).

Bank shards are REUSED from the 3138
full run (no recapture)."""
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
NAME = ('omega_p137_idinteract_'
        'portconsume_rewrite_xmat')
SMOKE = os.environ.get('P3139_SMOKE',
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
D38 = RDIR + r'\phase3138' \
      r'\omega_p136_statebank_' \
      r'bicdecomp_probecontrast'
MDIR_G = os.path.join(ROOT, 'models', 'hf',
                      'glm4-9b-chat-hf')
OUT = os.path.join(RDIR, 'phase3139', NAME)
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
CKPTF = os.path.join(OUT, 'p137_ckpt.pkl')
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
CINJ_L = (17, 26, 29, 32)
SCAN_N = 4 if SMOKE else 128
GEN_BATCH = 4 if SMOKE else 32
N_NEW = 12
ISH2_HARD = 0.75
ISH2_SOFT = 0.85
PORT_I_MIN = 0.25
PORT_C_MAX = 0.02
CINJ_QUIET = 0.10
XMAT_RETR_MIN = 0.30
RNG_SEED = 3139

# ---------------- frozen 3138 anchors --
EXP_V38 = ('a_3137_ok|i_sh_hard_ok|'
           'i_sh_soft85_below|c_sh_hard_ok|'
           'c_sh_soft85_ok|probe_auc_gen_ok|'
           'k0_anatomy_recorded|'
           'coverage_full')
RES38_SHA = 'f7ef08be'
# 6-decimal rounded anchors from
# p3138 result (compare tol 1e-6)
RETR38 = {'17': 1.0, '29': 0.784226,
          '33': 0.872024, '38': 0.956845}
ISH38_KEY17_P = 0.7134717106819153
CSH38_KEY17_MED = 0.9993661940097809
AUC_X_MED38 = 1.0
NS_I_17 = 0.13908881627533143
NS_B_17 = 0.971171470480975
# dvec sha8 (3135 npz, fp32)
DVEC_SHA = {17: '5e4c3085',
            29: 'ee9484b2',
            33: '59fbe0d3',
            38: 'aced803b'}
DVEC_MEDNORM = {17: 4.253485202789307,
                29: 31.643726348876953,
                33: 44.65449905395504,
                38: 100.42410278320312}
LEDGER_N = 275

# ---------------- seal ------------------
SEAL = {
    'phase': 3139,
    'name': NAME,
    'created': time.strftime(
        '%Y-%m-%d %H:%M:%S'),
    'smoke': SMOKE,
    'constants': {
        'NP': NP, 'N_T': N_T,
        'DIRS': list(DIRS),
        'ALL_L': ALL_L,
        'KEY_L': list(KEY_L),
        'CINJ_L': list(CINJ_L),
        'SCAN_N': SCAN_N,
        'GEN_BATCH': GEN_BATCH,
        'N_NEW': N_NEW,
        'ISH2_HARD': ISH2_HARD,
        'ISH2_SOFT': ISH2_SOFT,
        'PORT_I_MIN': PORT_I_MIN,
        'PORT_C_MAX': PORT_C_MAX,
        'CINJ_QUIET': CINJ_QUIET,
        'XMAT_RETR_MIN': XMAT_RETR_MIN,
        'RNG_SEED': RNG_SEED,
        'BANK_REUSE_3138': True},
    'anchors': {
        'res38_verdict': EXP_V38,
        'res38_sha8': RES38_SHA,
        'retr38': RETR38,
        'ish38_key17_P': ISH38_KEY17_P,
        'csh38_key17_med': CSH38_KEY17_MED,
        'auc_x_med38': AUC_X_MED38,
        'dvec_sha8': {str(k): v for k, v
                      in DVEC_SHA.items()},
        'ledger_n': LEDGER_N},
    'prereg': ('3138 closeout task 82 + '
               'FINGERPRINT_PARADIGM_PLAN '
               'Omega-P137: (1) interaction '
               'split ish2 + R row retrieval; '
               '(2) port consumption dvec '
               'B/C/I projection energies; '
               '(3) rewrite-window C(T2) vs '
               'displaced-I injection at '
               'L17/26/29/32 allstep; (4) '
               'cross-material 84 unseen '
               'pairs same-subject retrieval '
               '+ P/A1 AUC. Bank reused '
               'from 3138 full run. '
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
res38 = json.load(io.open(
    D38 + r'\result.json', encoding='utf-8'))
assert res38['smoke'] is False
led = json.load(io.open(
    ROOT + r'\research\gpt5\atlas'
    r'\atlas_ledger.json',
    encoding='utf-8'))
assert len(led['measurements']) == LEDGER_N
log('frozen inputs ok (26/35/38 + ledger '
    'n=%d)' % LEDGER_N)

# ================================================================
# PART A: 3138 link asserts
# ================================================================
log('== PART A: 3138 link asserts ==')
V38 = res38['verdict']
assert V38 == EXP_V38, V38
raw38 = io.open(
    D38 + r'\result.json', 'rb').read()
sha38 = hashlib.sha256(raw38).hexdigest()[:8]
assert sha38 == RES38_SHA, sha38
pc38 = res38['part_c']
pd38 = res38['part_d']
for l, v in RETR38.items():
    got = float(pc38['retr'][l])
    assert abs(got - v) < 1e-6, (l, got, v)
assert abs(float(pc38['ish']['17']['P'])
           - ISH38_KEY17_P) < 1e-9
_c17 = [float(v) for v
        in pc38['csh']['17'].values()]
assert abs(float(np.median(_c17))
           - CSH38_KEY17_MED) < 1e-9
assert abs(float(pd38['auc_x_med'])
           - AUC_X_MED38) < 1e-9
assert abs(float(
    pc38['norm_share']['17']['I'])
    - NS_I_17) < 1e-6
assert abs(float(
    pc38['norm_share']['17']['B'])
    - NS_B_17) < 1e-6
log('A hard asserts ok (3138 sha8 %s, '
    'retr/ish/csh/auc/normshare anchors)'
    % sha38)

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
# PART C: interaction split
# ================================================================
log('== PART C: interaction split ==')
rng_c = _rnd.Random(RNG_SEED)
row_half = np.array(
    [0] * (NP // 2) + [1] * (NP - NP // 2),
    dtype=np.int64)
rng_c.shuffle(row_half)

C_out = {'ish2': {}, 'r_retr': {},
         'i_templ_share': {},
         'norm_share': {}}
IDEINT_P = {}
CVEC = {}
for l in ALL_L:
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
    WR = X - B[None, None, None, :] \
        - Ideint[:, None, :, :] \
        - C[None, :, None, :]
    if l == 17:
        IDEINT_P[17] = Ideint[0].copy()
        CVEC[17] = C.copy()
    # ish2: identity stability across
    # template groups T01 vs T23, SAME
    # row set, global decomposition base
    # (B + group template mean removed;
    # stricter than 3138 in-half base)
    ish2_l = {}
    for di, dc in enumerate(DIRS):
        IA = X[di, :2].mean(axis=0) \
            - B[None, :] \
            - C[:2].mean(axis=0)[None, :]
        IB = X[di, 2:].mean(axis=0) \
            - B[None, :] \
            - C[2:].mean(axis=0)[None, :]
        cs = np.sum(IA * IB, axis=1) / (
            np.linalg.norm(IA, axis=1)
            * np.linalg.norm(IB, axis=1)
            + 1e-12)
        ish2_l[dc] = float(np.median(cs))
    C_out['ish2'][l] = ish2_l
    # R row-identity retrieval: does the
    # interaction residual carry row info?
    # SAME row set, two template groups.
    retr_l = []
    for di, dc in enumerate(DIRS):
        RA = WR[di, :2].mean(axis=0)
        RB = WR[di, 2:].mean(axis=0)
        RA = RA - RA.mean(axis=0)[None, :]
        RB = RB - RB.mean(axis=0)[None, :]
        RA_n = RA / (np.linalg.norm(
            RA, axis=1, keepdims=True)
            + 1e-12)
        RB_n = RB / (np.linalg.norm(
            RB, axis=1, keepdims=True)
            + 1e-12)
        sim = RA_n @ RB_n.T
        # RA/RB are the SAME row set under
        # two template groups (T01 vs T23):
        # correct top-1 target = arange(NP)
        retr_l.append(float(np.mean(
            np.argmax(sim, axis=1)
            == np.arange(NP))))
    C_out['r_retr'][l] = float(
        np.median(retr_l))
    # I_templ share: identity energy
    # explained by the template-direction
    # span {C[t] - Cbar}
    dC = C - C.mean(axis=0)[None, :]
    Uc, _Sc, _Vt = np.linalg.svd(
        dC.T, full_matrices=False)
    Qc = Uc[:, :3]
    It = Ideint.reshape(-1, HIDG)
    proj = (It @ Qc) @ Qc.T
    C_out['i_templ_share'][l] = float(
        np.median(
            np.sum(proj ** 2, axis=1)
            / (np.sum(It ** 2, axis=1)
               + 1e-12)))
    nB = float(np.linalg.norm(B))
    nI = float(np.median(np.linalg.norm(
        Ideint.reshape(-1, HIDG), axis=1)))
    nR = float(np.median(np.linalg.norm(
        WR.reshape(-1, HIDG), axis=1)))
    nH = float(np.median(np.linalg.norm(
        X.reshape(-1, HIDG), axis=1)))
    C_out['norm_share'][l] = {
        'B': nB / nH, 'I': nI / nH,
        'R': nR / nH}
    del X, B, Ideint, C, WR, It, proj
log('C interaction split done (40 layers)')

ish2_key = {dc: [C_out['ish2'][l][dc]
                 for l in KEY_L]
            for dc in DIRS}
r_retr_key = [C_out['r_retr'][l]
              for l in KEY_L]
itk_key = [C_out['i_templ_share'][l]
           for l in KEY_L]
ish2_all_med = float(np.median(
    [v for l in ALL_L
     for v in C_out['ish2'][l].values()]))
ish2_hard_ok = all(
    v >= ISH2_HARD
    for l in KEY_L
    for v in C_out['ish2'][l].values())
ish2_soft_ok = all(
    v >= ISH2_SOFT
    for l in KEY_L
    for v in C_out['ish2'][l].values())
r_retr_chance = 1.0 / NP
r_retr_med = float(np.median(r_retr_key))
r_above = r_retr_med >= 10 * r_retr_chance
log('C-GATE: ish2_key %s (hard %s soft %s); '
    'r_retr_key %s vs chance %.5f '
    '(above10x=%s); i_templ_key %s'
    % (json.dumps(ish2_key), ish2_hard_ok,
       ish2_soft_ok,
       json.dumps([round(v, 4)
                   for v in r_retr_key]),
       r_retr_chance, r_above,
       json.dumps([round(v, 4)
                   for v in itk_key])))
# prediction-type gates: record only
# (no hard assert -- a failed prediction
# is a verdict tag, not a crash; 3134
# lesson)

# ================================================================
# PART D: port consumption test
# ================================================================
log('== PART D: port consumption ==')
# decompose over KEY_L plus CINJ_L
# (PART E needs CVEC/IDEINT_P at the
# injection layers 26/32 too)
PORT = {}
for l in sorted(set(KEY_L)
                | set(CINJ_L)):
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
    if l not in KEY_L:
        del X, B, Ideint, C
        continue
    # Gram-Schmidt: qB -> C-perp -> I-perp
    qB = B / (np.linalg.norm(B) + 1e-12)
    Cperp = C - np.outer(
        C @ qB, qB)
    Q_C = Cperp / (np.linalg.norm(
        Cperp, axis=1, keepdims=True)
        + 1e-12)
    Ip = Ideint[0] - np.outer(
        Ideint[0] @ qB, qB) \
        - (Ideint[0] @ Q_C.T) @ Q_C
    Q_I, _R = np.linalg.qr(Ip.T.astype(
        np.float64))
    Q_I = Q_I[:, :Ip.shape[0]]
    D_ = dvec[l]
    dn2 = np.sum(D_ ** 2, axis=1)
    e_B = (D_ @ qB) ** 2 / dn2
    e_C = np.sum(
        (D_ @ Q_C.T) ** 2, axis=1) / dn2
    projI = (D_ @ Q_I) @ Q_I.T
    e_I = np.sum(projI ** 2,
                 axis=1) / dn2
    Iown = Ideint[0] / (np.linalg.norm(
        Ideint[0], axis=1,
        keepdims=True) + 1e-12)
    e_own = np.sum(
        D_ * Iown, axis=1) ** 2 / dn2
    e_res = 1.0 - e_B - e_C - e_I
    PORT[l] = {
        'e_B_med': float(np.median(e_B)),
        'e_C_med': float(np.median(e_C)),
        'e_I_med': float(np.median(e_I)),
        'e_own_med': float(np.median(e_own)),
        'e_res_med': float(np.median(e_res)),
        'own_over_I': float(np.median(
            e_own / np.maximum(e_I, 1e-12)))}
    log('D port L%02d: B %.4f | C %.4f | '
        'I %.4f (own %.4f) | resid %.4f'
        % (l, PORT[l]['e_B_med'],
           PORT[l]['e_C_med'],
           PORT[l]['e_I_med'],
           PORT[l]['e_own_med'],
           PORT[l]['e_res_med']))
    del X, B, Ideint, C, Ip, Q_I, projI
port_I_med = float(np.median(
    [PORT[l]['e_I_med'] for l in KEY_L]))
port_C_med = float(np.median(
    [PORT[l]['e_C_med'] for l in KEY_L]))
port_I_ok = port_I_med >= PORT_I_MIN
port_c_ok = port_C_med <= PORT_C_MAX
log('D-SOFT: port_I_med=%.4f (>= %.2f: %s); '
    'port_C_med=%.4f (<= %.3f: %s)'
    % (port_I_med, PORT_I_MIN, port_I_ok,
       port_C_med, PORT_C_MAX, port_c_ok))
# prediction-type gates: record only

# ================================================================
# PART E: rewrite-window causality
# ================================================================
log('== PART E: C/I injection trials ==')
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
    gen_base_E = _EB['gens']
    xphase_E = _EB['xphase']
    log('E base RESUMED (xphase=%.4f)'
        % xphase_E)
else:
    gen_base_E = []
    for b0 in range(0, SCAN_N, GEN_BATCH):
        gen_base_E.extend(gen_batch_g2(
            rows_scan[b0:b0 + GEN_BATCH]))
    _xp = [pad12(gen_base_E[j]) ==
           pad12([int(v) for v in
                  z26['gen_base_P'][j]])
           for j in range(SCAN_N)]
    xphase_E = float(np.mean(_xp))
    log('E xphase record: session-base vs '
        'z26 bit-match %.4f (%d/%d)'
        % (xphase_E, int(np.sum(_xp)),
           SCAN_N))
    ck_save('e_base', {
        'gens': gen_base_E,
        'xphase': xphase_E})
base12_E = [pad12(gen_base_E[j])
            for j in range(SCAN_N)]

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
        trial_metrics(gen_l, base12_E)
    E_res[tname] = {'chg': chg_l,
                    'first': int(first_l)}
    log('%s: chg=%.4f first=%d'
        % (tname, chg_l, first_l))
    ck_save(tname, {'res': E_res[tname],
                    'fstep': fstep})


# C(T2) injections: does the template
# component drive behavior?
for il in CINJ_L:
    _cv = CVEC[il][2][None, :]
    _dv = np.repeat(_cv, SCAN_N, axis=0) \
        .astype(np.float32)
    _run_inj_trial('cinj_l%02d_d2.0' % il,
                   il, _dv, 2.0, 'allstep')
_c29 = CVEC[29][2][None, :]
_dv29 = np.repeat(_c29, SCAN_N,
                  axis=0).astype(np.float32)
_run_inj_trial('cinj_l29_d1.0', 29, _dv29,
               1.0, 'allstep')
# displaced-identity injections:
# row i receives identity i+1
for il in CINJ_L:
    _iv = np.roll(IDEINT_P[il], -1,
                  axis=0)[:SCAN_N] \
        .astype(np.float32)
    _run_inj_trial('iinj_l%02d_d1.0' % il,
                   il, _iv, 1.0, 'allstep')
cinj_chgs = [E_res[k]['chg'] for k in E_res
             if k.startswith('cinj')]
iinj_chgs = [E_res[k]['chg'] for k in E_res
             if k.startswith('iinj')]
cinj_quiet = all(
    c <= CINJ_QUIET for c in cinj_chgs)
iinj_max = float(max(iinj_chgs))
log('E-SOFT: cinj chgs %s (quiet<=%.2f: '
    '%s); iinj chgs %s (max %.4f)'
    % (json.dumps([round(c, 4)
                   for c in cinj_chgs]),
       CINJ_QUIET, cinj_quiet,
       json.dumps([round(c, 4)
                   for c in iinj_chgs]),
       iinj_max))

# ================================================================
# PART F: cross-material sentinel
# ================================================================
log('== PART F: cross-material ==')
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
    """New-pair prompt: same facts style,
    distractor lines sampled with fixed
    seed (query line first, k=0)."""
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


rows_x = {dc: [] for dc in DIRS}
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
    for dc in DIRS:
        if dc == 'P':
            (qrel, lrel) = (rp, rp)
        else:
            rp2 = None
            for rr in range(
                    len(PREDS_all)):
                if rr not in used_rels \
                        and rr != rp:
                    rp2 = rr
                    break
            (qrel, lrel) = (rp2, rp)
        txt = build_prompt_new(
            s, o, lrel, qrel)
        rows_x[dc].append(list(tok_g(
            txt, add_special_tokens=False)
            ['input_ids']))
_XK = CK['data'].get('xmat_cap')
if _XK is not None:
    xst = {int(k): v for k, v in
           _XK['states'].items()}
    log('F capture RESUMED')
else:
    xst = capture_states(
        rows_x['P'] + rows_x['A1'],
        [17, 29, 38])
    ck_save('xmat_cap', {
        'states': {str(k): v
                   for k, v in xst.items()}})
xp = {l: xst[l][:XMAT_N]
      for l in xst}
xa = {l: xst[l][XMAT_N:]
      for l in xst}
# same-subject retrieval: new-P hidden vs
# 672-row identity bank (I_deint P)
retr_s = {}
retr_pair = {}
for l in (17, 29, 38):
    bank = IDEINT_P[l]
    bank_n = bank / (np.linalg.norm(
        bank, axis=1, keepdims=True)
        + 1e-12)
    q = xp[l]
    q_n = q / (np.linalg.norm(
        q, axis=1, keepdims=True) + 1e-12)
    sim = q_n @ bank_n.T
    top1 = np.argmax(sim, axis=1)
    hit_s = 0
    hit_pair = 0
    for i, pk in enumerate(new_pairs):
        (s, o) = (int(v)
                  for v in pk.split('_'))
        j = int(top1[i])
        (s2, o2) = (int(v)
                    for v in
                    pks[j].split('_'))
        hit_s += int(s2 == s)
        hit_pair += int(s2 == s and o2 == o)
    retr_s[l] = hit_s / XMAT_N
    retr_pair[l] = hit_pair / XMAT_N
    log('F retr L%02d: same-s %.4f '
        'same-pair %.4f (chance-s ~%.4f)'
        % (l, retr_s[l], retr_pair[l],
           1.0 / ents_n))
# cross-material P/A1 AUC
from sklearn.linear_model import \
    LogisticRegression  # noqa: E402
from sklearn.metrics import roc_auc_score \
    # noqa: E402

auc_xm = {}
for l in (17, 29, 38):
    Xb = np.concatenate([
        H_ld['P'][0][:, l, :]
        * H_sc['P'][0][l],
        H_ld['A1'][0][:, l, :]
        * H_sc['A1'][0][l]])
    yb = np.concatenate([
        np.zeros(NP, dtype=np.int64),
        np.ones(NP, dtype=np.int64)])
    Xn = np.concatenate([xp[l], xa[l]])
    yn = np.concatenate([
        np.zeros(XMAT_N, dtype=np.int64),
        np.ones(XMAT_N, dtype=np.int64)])
    clf = LogisticRegression(
        max_iter=2000, C=1.0)
    clf.fit(Xb, yb)
    p = clf.predict_proba(Xn)[:, 1]
    auc_xm[l] = float(
        roc_auc_score(yn, p))
    log('F xmat AUC L%02d = %.4f'
        % (l, auc_xm[l]))
retr_s_med = float(np.median(
    [retr_s[l] for l in (17, 29, 38)]))
xmat_ok = retr_s_med >= XMAT_RETR_MIN
log('F-SOFT: retr_same_s med=%.4f '
    '(>= %.2f: %s); xmat AUC %s'
    % (retr_s_med, XMAT_RETR_MIN,
       xmat_ok,
       json.dumps({str(l): round(v, 4)
                   for l, v in
                   auc_xm.items()})))

# ================================================================
# verdict + result.json + npz
# ================================================================
tags = ['a_3138_ok',
        'ish2_hard75_ok' if ish2_hard_ok
        else 'ish2_hard75_fail',
        'ish2_soft85_ok' if ish2_soft_ok
        else 'ish2_soft85_below',
        'r_retr_above_chance' if r_above
        else 'r_retr_chance',
        'port_I_dominant' if port_I_ok
        else 'port_I_below',
        'port_c_near_zero' if port_c_ok
        else 'port_c_signal',
        'cinj_quiet' if cinj_quiet
        else 'cinj_active',
        'iinj_recorded',
        'xmat_retr_ok' if xmat_ok
        else 'xmat_retr_below',
        'xmat_auc_recorded',
        'coverage_full']
verdict = '|'.join(tags)
runtime = time.time() - T0
log('VERDICT: %s' % verdict)
log('DONE (%.1fs)' % runtime)

result = {
    'phase': 3139,
    'name': NAME,
    'smoke': SMOKE,
    'verdict': verdict,
    'runtime_s': runtime,
    'seal_sha8': hashlib.sha256(
        json.dumps(SEAL, sort_keys=True,
                   ensure_ascii=False)
        .encode('utf-8')).hexdigest()[:8],
    'part_c': {
        'ish2': {str(l): C_out['ish2'][l]
                 for l in ALL_L},
        'r_retr': {str(l): C_out['r_retr'][l]
                   for l in ALL_L},
        'i_templ_share': {
            str(l): C_out['i_templ_share'][l]
            for l in ALL_L},
        'norm_share': {
            str(l): C_out['norm_share'][l]
            for l in ALL_L},
        'ish2_all_med': ish2_all_med,
        'gates': {
            'ish2_hard75': ish2_hard_ok,
            'ish2_soft85': ish2_soft_ok,
            'r_retr_above10x': r_above}},
    'part_d': {
        'port': {str(l): PORT[l]
                 for l in KEY_L},
        'port_I_med': port_I_med,
        'port_C_med': port_C_med},
    'part_e': {
        'xphase': xphase_E,
        'trials': E_res,
        'cinj_quiet': cinj_quiet,
        'iinj_max': iinj_max},
    'part_f': {
        'n_pairs': XMAT_N,
        'retr_s': {str(l): retr_s[l]
                   for l in (17, 29, 38)},
        'retr_pair': {str(l): retr_pair[l]
                      for l in (17, 29, 38)},
        'auc_xmat': {str(l): auc_xm[l]
                     for l in (17, 29, 38)},
        'retr_s_med': retr_s_med}}
with io.open(os.path.join(
        OUT, 'result.json'), 'w',
        encoding='utf-8') as f:
    json.dump(result, f,
              ensure_ascii=False, indent=1)
np.savez(os.path.join(
    OUT, 'p137_readout.npz'),
    row_half=row_half,
    ish2_med=np.array(
        [np.median(list(
            C_out['ish2'][l].values()))
         for l in ALL_L]),
    r_retr=np.array(
        [C_out['r_retr'][l]
         for l in ALL_L]),
    i_templ_share=np.array(
        [C_out['i_templ_share'][l]
         for l in ALL_L]),
    port_eB=np.array(
        [PORT[l]['e_B_med']
         for l in KEY_L]),
    port_eC=np.array(
        [PORT[l]['e_C_med']
         for l in KEY_L]),
    port_eI=np.array(
        [PORT[l]['e_I_med']
         for l in KEY_L]),
    port_eown=np.array(
        [PORT[l]['e_own_med']
         for l in KEY_L]),
    port_eres=np.array(
        [PORT[l]['e_res_med']
         for l in KEY_L]),
    xmat_retr_s=np.array(
        [retr_s[l] for l in (17, 29, 38)]),
    xmat_auc=np.array(
        [auc_xm[l] for l in (17, 29, 38)]))
log('dumps done: result.json + seal + '
    'p137_readout.npz')
if not SMOKE:
    if os.path.exists(CKPTF):
        os.remove(CKPTF)
        log('CKPT removed (data fully '
            'persisted)')
print('P3139 DONE (%.1fs)' % runtime)
