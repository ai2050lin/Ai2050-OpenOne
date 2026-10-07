# -*- coding: utf-8 -*-
"""Phase 3138 (Omega-P136): Absolute State
Bank + B/I/C additive decomposition +
probe contrast + k0 row-level anatomy.

Preregistered in 3137 closeout (task 81,
aligned with FINGERPRINT_PARADIGM_PLAN.md
Omega-P136):
(1) state bank: BANK_N rows x 2
directions (P/A1) x 4 surface templates
(T0 standard / T1 line-reorder / T2
prefix / T3 query wording) x all 40
layers, last-position states, fp16
sharded npz (<=1.8 GB);
(2) per-layer additive fit H = B + I
+ C + R: norm shares; per-row identity
stability across template halves (T01
vs T23, in-half base) = the invariance
gate; template-component stability
across row halves (own-half base);
cross-half top-1 identity retrieval;
(3) P-vs-A1 logistic contrast: raw H vs
B+I vs I-only, two regimes (within-T0
row split and cross-template T01->T23);
C-only is chance by construction;
(4) k0 anatomy offline from 3136 npz +
z26 margins: only_nok0 vs only_full
flip rows base-margin compare.

Capture rows are the plain prompt token
sequences (pids s0 convention, 3133-3137
lineage); template variants change ONLY
the prompt surface (same entities /
relations / query truth), generation
segment untouched."""
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
NAME = ('omega_p136_statebank_'
        'bicdecomp_probecontrast')
SMOKE = os.environ.get('P3138_SMOKE',
                       '') == '1'

D05 = os.path.join(RDIR, 'phase3105',
                   'omega_p103_incontext_truth_'
                   'consistency')
D26 = RDIR + r'\phase3126' \
      r'\omega_p124_glm4_anchoredlast_' \
      r'regen_writechain'
D32 = RDIR + r'\phase3132' \
      r'\omega_p130_forkcausal_single256'
D35 = RDIR + r'\phase3135' \
      r'\omega_p133_conduction_' \
      r'co36ablation_window'
D36 = RDIR + r'\phase3136' \
      r'\omega_p134_conddose_' \
      r'crossmatrix_w8drop'
D37 = RDIR + r'\phase3137' \
      r'\omega_p135_modecoop_' \
      r'k0anat_coorddecomp_l35recheck'
MDIR_G = os.path.join(ROOT, 'models', 'hf',
                      'glm4-9b-chat-hf')
OUT = os.path.join(RDIR, 'phase3138', NAME)
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
CKPTF = os.path.join(OUT, 'p136_ckpt.pkl')
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
BANK_N = 4 if SMOKE else NP
N_T = 4
DIRS = ('P', 'A1')
ALL_L = list(range(40))
L17 = 17
KEY_L = (17, 29, 33, 38)
# gates (hard floors kept low for
# long-run safety; 0.85 targets are
# recorded as soft tags)
ISH_HARD = 0.60
CSH_HARD = 0.60
ISH_SOFT = 0.85
CSH_SOFT = 0.85
AUC_GEN = 0.65
RNG_SEED = 3138
TF_SEED = 3138

# ---------------- frozen 3137 anchors --
EXP_V37 = ('a_3136_ok|mode_gap_all|'
           'po_dose_monotone|'
           'k0_negative_confirmed|'
           'k0_alone_active|'
           'co36_l17_flat|'
           'coord_dose_monotone|'
           'l35_union_zero_dose_robust|'
           'coverage_full')
RES37_SHA = '4f8055bb'
CO36_SHA = '84a1e1a3'
CO36_RANK37_SHA = '0f4c25b1'
CO50_SHA = '52b126af'
UNION_SHA = '1e7edb1a'
K0ONLY_SHA = '9ba70f6f'
# 3137 hard re-asserts (drift-0 values)
A37 = {'all_l29_s2.0': 0.640625,
       'all_l33_s2.0': 0.515625,
       'all_l38_s2.0': 0.9453125,
       'a672': 0.0892857142857143,
       'w8full': 0.6484375,
       'drop3': 0.40625,
       'g_co36_d1': 0.140625,
       'g_co50_d1': 0.421875,
       'h_l35union_d1': 0.0546875}
LEDGER_N = 274

# ---------------- seal ------------------
SEAL = {
    'phase': 3138,
    'name': NAME,
    'created': time.strftime(
        '%Y-%m-%d %H:%M:%S'),
    'smoke': SMOKE,
    'constants': {
        'NP': NP, 'BANK_N': BANK_N,
        'N_T': N_T, 'DIRS': list(DIRS),
        'ALL_L': ALL_L, 'L17': L17,
        'KEY_L': list(KEY_L),
        'ISH_HARD': ISH_HARD,
        'CSH_HARD': CSH_HARD,
        'ISH_SOFT': ISH_SOFT,
        'CSH_SOFT': CSH_SOFT,
        'AUC_GEN': AUC_GEN,
        'TF_SEED': TF_SEED,
        'RNG_SEED': RNG_SEED,
        'BANK_FP16_SCALED': True},
    'anchors': {
        'res37_verdict': EXP_V37,
        'res37_sha8': RES37_SHA,
        'co36_sha8': CO36_SHA,
        'co36_rank37_sha8': CO36_RANK37_SHA,
        'co50_sha8': CO50_SHA,
        'union_sha8': UNION_SHA,
        'k0only_sha8': K0ONLY_SHA,
        'a37_hard': A37,
        'ledger_n': LEDGER_N},
    'prereg': ('3137 closeout task 81 + '
               'FINGERPRINT_PARADIGM_PLAN '
               'Omega-P136: (1) state bank '
               'BANK_N x 2 dirs x 4 templates '
               'x 40 layers fp16 sharded; '
               '(2) per-layer additive fit '
               'B+I+C+R, norm shares, '
               'per-row identity stability '
               'T01 vs T23, template '
               'stability across row halves, '
               'cross-half top-1 retrieval; '
               '(3) P-vs-A1 logistic raw vs '
               'BI vs I, within-T0 and '
               'cross-template; (4) k0 '
               'row-level anatomy offline. '
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
assert z26['gen_base_A1'].shape == (672, 12)
z32 = np.load(D32 + r'\p130_readout.npz',
              allow_pickle=False)
assert z32['co50'].shape == (50,)
z35 = np.load(D35 + r'\p133_readout.npz',
              allow_pickle=False)
for k, sha in (('co36', CO36_SHA),
               ('co36_rank',
                CO36_RANK37_SHA)):
    a = z35[k]
    got = hashlib.sha256(
        a.tobytes()).hexdigest()[:8]
    assert got == sha, (k, got, sha)
co50 = z32['co50'].astype(np.int64)
assert hashlib.sha256(
    co50.tobytes()).hexdigest()[:8] \
    == CO50_SHA
union3650 = np.union1d(
    z35['co36'].astype(np.int64), co50)
assert hashlib.sha256(
    union3650.tobytes()).hexdigest()[:8] \
    == UNION_SHA
assert len(union3650) == 100
z37 = np.load(D37 + r'\p135_readout.npz',
              allow_pickle=False)
k0s = hashlib.sha256(
    z37['fstep_w8_k0only'].tobytes()
).hexdigest()[:8]
po17s = hashlib.sha256(
    z37['fstep_po_l17_s1.0'].tobytes()
).hexdigest()[:8]
assert k0s == K0ONLY_SHA, k0s
assert k0s == po17s, \
    'k0only != po_l17_s1.0'
res37 = json.load(io.open(
    D37 + r'\result.json', encoding='utf-8'))
assert res37['smoke'] is False
led = json.load(io.open(
    ROOT + r'\research\gpt5\atlas'
    r'\atlas_ledger.json',
    encoding='utf-8'))
assert len(led['measurements']) == LEDGER_N
log('frozen inputs ok (26/32/35/37 + '
    'ledger n=%d)' % LEDGER_N)

# ================================================================
# PART A: offline 3137 link asserts
# ================================================================
log('== PART A: 3137 link asserts ==')
V37 = res37['verdict']
assert V37 == EXP_V37, V37
raw37 = io.open(
    D37 + r'\result.json', 'rb').read()
sha37 = hashlib.sha256(raw37).hexdigest()[:8]
assert sha37 == RES37_SHA, sha37
pe37 = res37['part_e']
pf37 = res37['part_f']
pg37 = res37['part_g']
ph37 = res37['part_h']
mm37 = pe37['mode_matrix']
assert abs(mm37['all_l29_s2.0']['chg']
           - A37['all_l29_s2.0']) < 1e-9
assert abs(mm37['all_l33_s2.0']['chg']
           - A37['all_l33_s2.0']) < 1e-9
assert abs(mm37['all_l38_s2.0']['chg']
           - A37['all_l38_s2.0']) < 1e-9
assert abs(pe37['a672_l38_po_s2']['chg']
           - A37['a672']) < 1e-9
fr37 = pf37['f_res']
assert abs(fr37['w8full']['chg']
           - A37['w8full']) < 1e-9
assert abs(fr37['drop3']['chg']
           - A37['drop3']) < 1e-9
assert abs(pg37['g_res']['g_co36_d1']['chg']
           - A37['g_co36_d1']) < 1e-9
assert abs(pg37['g_res']['g_co50_d1']['chg']
           - A37['g_co50_d1']) < 1e-9
assert abs(ph37['h_res']['h_l35union_d1']
           ['chg'] - A37['h_l35union_d1']) \
    < 1e-9
log('A hard asserts ok (3137 sha8 %s, '
    '9 drift-0 anchors re-asserted)'
    % sha37)

# ================================================================
# materials factory (3125/.../3137
# identical, + template code)
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


def build_prompt_tc(mat, s, o, lrel, qrel,
                    tcode):
    ents = mat['entities']
    PREDS = mat['predicates']
    D = [tuple(d) for d in
         mat['distractors']['%d_%d' % (s, o)]]
    k = mat['kline']['%d_%d' % (s, o)]
    lines = [(s, lrel, o)] + list(D)
    _ordtag = 'ord5' if tcode != 1 \
        else 'ord6'
    rng2 = _rnd.Random(zlib.crc32(
        ('%d_%d_%s' % (s, o, _ordtag))
        .encode('ascii')))
    order = list(range(8))
    rng2.shuffle(order)
    lines = [lines[i] for i in order]
    ci = lines.index((s, lrel, o))
    lines[ci], lines[k] = lines[k], lines[ci]
    if tcode == 2:
        text = 'Given facts:'
    else:
        text = 'Facts:'
    for (ls, lr, lo) in lines:
        text += ' The %s %s the %s.' % (
            ents[ls], PREDS[lr], ents[lo])
    if tcode == 3:
        text += (' Query: The %s %s the %s. '
                 'True or false? Answer:'
                 % (ents[s], PREDS[qrel],
                    ents[o]))
    else:
        text += (' Query: The %s %s the %s. '
                 'Is this query true? Answer:'
                 % (ents[s], PREDS[qrel],
                    ents[o]))
    return text


TEMPLATE_DESC = {
    0: 'T0 standard (Facts:/ord5/Is this '
       'query true?)',
    1: 'T1 line-reorder (ord6 seed)',
    2: 'T2 prefix (Given facts:)',
    3: 'T3 query wording (True or '
       'false?)'}
log('factory defined: %s'
    % json.dumps(TEMPLATE_DESC))

# ================================================================
# model load
# ================================================================
log('== load glm4-9b ==')
import torch  # noqa: E402
from transformers import AutoModelForCausalLM, \
    AutoTokenizer  # noqa: E402

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
tok_g.padding_side = 'left'


def capture_states(rows_all, cap_layers):
    """Last-position states at cap_layers.
    SINGLE-SAMPLE loop (batch=1, zero
    padding) replicating z26/3131
    semantics exactly (3133 rev-f)."""
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


# ================================================================
# PART B: Absolute State Bank
# ================================================================
log('== PART B: state bank capture ==')
rows_T = {}
lens_T = {}
for dc in DIRS:
    rows_T[dc] = {}
    lens_T[dc] = {}
    for t in range(N_T):
        ids_l = []
        for j in range(BANK_N):
            pk = pks[j]
            (s, o) = (int(v)
                      for v in pk.split('_'))
            r = p2r[pk]
            ri1, ri2 = frel[pk]
            if dc == 'P':
                (qrel, lrel) = (r, r)
            else:
                (qrel, lrel) = (r, ri1)
            txt = build_prompt_tc(
                mat5, s, o, lrel, qrel, t)
            ids_l.append(list(tok_g(
                txt,
                add_special_tokens=False)
                ['input_ids']))
        rows_T[dc][t] = ids_l
        lens_T[dc][t] = [len(x)
                         for x in ids_l]
        log('rows %s T%d built (len med %d)'
            % (dc, t, int(np.median(
                lens_T[dc][t]))))

bank_ok = {}
for dc in DIRS:
    for t in range(N_T):
        tag = '%s_T%d' % (dc, t)
        fp = os.path.join(
            OUT, 'bank_%s.npz' % tag)
        done = False
        if os.path.exists(fp):
            try:
                zt = np.load(fp,
                             allow_pickle=False)
                if 'scale' in zt.files \
                        and zt['states'].shape == \
                        (BANK_N, 40, HIDG):
                    done = True
            except Exception:
                done = False
        if done:
            bank_ok[tag] = fp
            log('bank %s EXISTS, skip' % tag)
            continue
        st = capture_states(
            rows_T[dc][t], ALL_L)
        arr = np.stack(
            [st[l] for l in ALL_L],
            axis=1)
        amax = np.abs(arr).max(
            axis=(0, 2))
        scale = np.maximum(
            1.0, amax / 6e4) \
            .astype(np.float32)
        arr16 = (arr
                 / scale[None, :, None]
                 ).astype(np.float16)
        np.savez(fp,
                 states=arr16,
                 scale=scale,
                 lens=np.array(
                     lens_T[dc][t],
                     dtype=np.int32))
        bank_ok[tag] = fp
        log('bank %s saved %s (fp16)'
            % (tag, str(arr.shape)))
        del st, arr
        gc.collect()
ck_save('bank_done', {
    'tags': sorted(bank_ok.keys())})
assert len(bank_ok) == 2 * N_T

# ================================================================
# PART C: B/I/C additive fit
# ================================================================
log('== PART C: B/I/C decomposition ==')
H_ld = {}
H_sc = {}
for dc in DIRS:
    H_ld[dc] = {}
    H_sc[dc] = {}
    for t in range(N_T):
        zt = np.load(bank_ok['%s_T%d'
                             % (dc, t)],
                     allow_pickle=False)
        _a = zt['states'] \
            .astype(np.float32)
        assert np.isfinite(_a).all(), \
            'bank shard non-finite'
        H_ld[dc][t] = _a
        H_sc[dc][t] = zt['scale'] \
            .astype(np.float32)
        assert H_ld[dc][t].shape == \
            (BANK_N, 40, HIDG)
log('bank loaded (8 shards fp16)')

rng_c = _rnd.Random(RNG_SEED)
row_half = np.array(
    [0] * (BANK_N // 2)
    + [1] * (BANK_N - BANK_N // 2),
    dtype=np.int64)
rng_c.shuffle(row_half)

C_out = {
    'norm_share': {}, 'ish': {},
    'csh': {}, 'retr': {}, 'resid': {}}
for l in ALL_L:
    X = np.zeros((2, N_T, BANK_N, HIDG),
                 dtype=np.float32)
    for di, dc in enumerate(DIRS):
        for t in range(N_T):
            X[di, t] = \
                H_ld[dc][t][:, l, :] \
                * H_sc[dc][t][l]
    B = X.mean(axis=(0, 1, 2))
    I = X.mean(axis=1) - B
    C = X.mean(axis=(0, 2)) - B
    R = X - B[None, None, None, :] \
        - I[:, None, :, :] \
        - C[None, :, None, :]
    nB = float(np.linalg.norm(B))
    nI = float(np.median(np.linalg.norm(
        I.reshape(-1, HIDG), axis=1)))
    nC = float(np.median(np.linalg.norm(
        C, axis=1)))
    nR = float(np.median(np.linalg.norm(
        R.reshape(-1, HIDG), axis=1)))
    nH = float(np.median(np.linalg.norm(
        X.reshape(-1, HIDG), axis=1)))
    C_out['norm_share'][l] = {
        'B': nB / nH, 'I': nI / nH,
        'C': nC / nH, 'R': nR / nH,
        'absB': nB, 'absI': nI,
        'absC': nC, 'absR': nR}
    # per-row identity stability across
    # template halves + cross-half top-1
    # retrieval (same in-half-based feats)
    ish_l = {}
    retr_l = []
    for di, dc in enumerate(DIRS):
        XA = X[di, :2].mean(axis=0)
        XB = X[di, 2:].mean(axis=0)
        IA = XA - XA.mean(axis=0)[None, :]
        IB = XB - XB.mean(axis=0)[None, :]
        cs = np.sum(IA * IB, axis=1) / (
            np.linalg.norm(IA, axis=1)
            * np.linalg.norm(IB, axis=1)
            + 1e-12)
        ish_l[dc] = float(np.median(cs))
        IA_n = IA / (np.linalg.norm(
            IA, axis=1, keepdims=True)
            + 1e-12)
        IB_n = IB / (np.linalg.norm(
            IB, axis=1, keepdims=True)
            + 1e-12)
        sim = IA_n @ IB_n.T
        retr_l.append(float(np.mean(
            np.argmax(sim, axis=1)
            == np.arange(BANK_N))))
    C_out['ish'][l] = ish_l
    C_out['retr'][l] = float(
        np.median(retr_l))
    # template-component stability across
    # row halves (each half uses its OWN
    # internal base: no leakage)
    BgA = X[:, :, row_half == 0] \
        .mean(axis=(0, 1, 2))
    BgB = X[:, :, row_half == 1] \
        .mean(axis=(0, 1, 2))
    csh_l = {}
    for t in range(N_T):
        CA = X[:, t][:, row_half == 0] \
            .mean(axis=(0, 1)) - BgA
        CB = X[:, t][:, row_half == 1] \
            .mean(axis=(0, 1)) - BgB
        csh_l[t] = float(np.dot(CA, CB) / (
            np.linalg.norm(CA)
            * np.linalg.norm(CB) + 1e-12))
    C_out['csh'][l] = csh_l
    # residual top-1 PCA share
    # (descriptive)
    Rf = R.reshape(-1, HIDG)
    Rf = Rf - Rf.mean(axis=0)[None, :]
    U, S, Vt = np.linalg.svd(
        Rf[::max(1, Rf.shape[0] // 512)],
        full_matrices=False)
    C_out['resid'][l] = float(
        S[0] ** 2 / max(1e-12,
                        (S ** 2).sum()))
    del X, B, I, C, R, Rf, U, S, Vt
log('C decomposition done (40 layers)')

ish_key = {dc: [C_out['ish'][l][dc]
                for l in KEY_L]
           for dc in DIRS}
csh_key = [float(np.median(
    [C_out['csh'][l][t]
     for t in range(N_T)]))
    for l in KEY_L]
retr_rand = 1.0 / BANK_N
retr_key = {l: C_out['retr'][l]
            for l in KEY_L}
ish_all_med = float(np.median(
    [v for l in ALL_L
     for v in C_out['ish'][l].values()]))
csh_all_med = float(np.median(
    [v for l in ALL_L
     for v in C_out['csh'][l].values()]))
ish_hard_ok = all(
    v >= ISH_HARD
    for l in KEY_L
    for v in C_out['ish'][l].values())
csh_hard_ok = all(
    v >= CSH_HARD
    for l in KEY_L
    for v in C_out['csh'][l].values())
ish_soft_ok = all(
    v >= ISH_SOFT
    for l in KEY_L
    for v in C_out['ish'][l].values())
csh_soft_ok = all(
    v >= CSH_SOFT
    for l in KEY_L
    for v in C_out['csh'][l].values())
log('C-GATE: ish_key %s (hard %s soft %s); '
    'csh_key %s (hard %s soft %s); '
    'retr_key %s vs rand %.4f'
    % (json.dumps(ish_key), ish_hard_ok,
       ish_soft_ok, json.dumps(csh_key),
       csh_hard_ok, csh_soft_ok,
       json.dumps(retr_key), retr_rand))
if not SMOKE:
    assert ish_hard_ok, json.dumps(ish_key)
    assert csh_hard_ok, \
        json.dumps(csh_key)

# ================================================================
# PART D: probe contrast (P vs A1)
# ================================================================
log('== PART D: probe contrast ==')
from sklearn.linear_model import \
    LogisticRegression  # noqa: E402
from sklearn.metrics import roc_auc_score \
    # noqa: E402

n_tr = (BANK_N * 2) // 3
row_perm = np.random.RandomState(
    RNG_SEED).permutation(BANK_N)
tr_rows = row_perm[:n_tr]
te_rows = row_perm[n_tr:]


def _auc(feat_tr, y_tr, feat_te, y_te):
    if len(np.unique(y_tr)) < 2 \
            or len(np.unique(y_te)) < 2:
        return float('nan')
    clf = LogisticRegression(
        max_iter=2000, C=1.0)
    clf.fit(feat_tr, y_tr)
    p = clf.predict_proba(feat_te)[:, 1]
    return float(roc_auc_score(y_te, p))


D_out = {'auc': {}, 'auc_x': {}}
for l in ALL_L:
    X = np.zeros((2, N_T, BANK_N, HIDG),
                 dtype=np.float32)
    for di, dc in enumerate(DIRS):
        for t in range(N_T):
            X[di, t] = \
                H_ld[dc][t][:, l, :] \
                * H_sc[dc][t][l]
    B = X.mean(axis=(0, 1, 2))
    I = X.mean(axis=1) - B
    C = X.mean(axis=(0, 2)) - B
    Hhat = B[None, None, None, :] \
        + I[:, None, :, :] \
        + C[None, :, None, :]
    I_full = np.broadcast_to(
        B[None, None, None, :]
        + I[:, None, :, :],
        (2, N_T, BANK_N, HIDG))
    comps = {
        'raw': X,
        'BI': Hhat,
        'I': I_full}
    a_l = {}
    ax_l = {}
    for cname, Xc in comps.items():
        # within-T0 regime: row split
        f_tr = np.concatenate(
            [Xc[0, 0][tr_rows],
             Xc[1, 0][tr_rows]])
        y_tr = np.concatenate(
            [np.zeros(n_tr, dtype=np.int64),
             np.ones(n_tr, dtype=np.int64)])
        f_te = np.concatenate(
            [Xc[0, 0][te_rows],
             Xc[1, 0][te_rows]])
        y_te = np.concatenate(
            [np.zeros(BANK_N - n_tr,
                      dtype=np.int64),
             np.ones(BANK_N - n_tr,
                     dtype=np.int64)])
        a_l[cname] = _auc(f_tr, y_tr,
                          f_te, y_te)
        # cross-template regime: train on
        # T01 mean feats, test on T23 mean
        # feats (row split both sides)
        f_tr2 = np.concatenate(
            [Xc[0, :2].mean(axis=0)[tr_rows],
             Xc[1, :2].mean(axis=0)[tr_rows]])
        f_te2 = np.concatenate(
            [Xc[0, 2:].mean(axis=0)[te_rows],
             Xc[1, 2:].mean(axis=0)[te_rows]])
        ax_l[cname] = _auc(f_tr2, y_tr,
                           f_te2, y_te)
    # C-only: no row info -> chance
    a_l['C_only'] = 0.5
    ax_l['C_only'] = 0.5
    D_out['auc'][l] = a_l
    D_out['auc_x'][l] = ax_l
    del X, B, I, C, Hhat, comps
log('D probe done (40 layers)')
raw_key = [D_out['auc_x'][l]['raw']
           for l in KEY_L]
log('D-SOFT: raw_x key AUC %s; I_x key %s; '
    'BI_x key %s'
    % (json.dumps([round(v, 4)
                   for v in raw_key]),
       json.dumps([round(
           D_out['auc_x'][l]['I'], 4)
           for l in KEY_L]),
       json.dumps([round(
           D_out['auc_x'][l]['BI'], 4)
           for l in KEY_L])))
auc_x_med = float(np.median(
    [D_out['auc_x'][l]['raw']
     for l in ALL_L]))
auc_gen_ok = auc_x_med >= AUC_GEN

# ================================================================
# PART E: k0 row-level anatomy (offline)
# ================================================================
log('== PART E: k0 row anatomy ==')
f_full = z37['fstep_w8full'] \
    .astype(np.int64)
f_nok0 = z37['fstep_w8_nok0'] \
    .astype(np.int64)
only_full = np.where(
    (f_full >= 0) & (f_nok0 < 0))[0]
only_nok0 = np.where(
    (f_nok0 >= 0) & (f_full < 0))[0]
both = np.where(
    (f_full >= 0) & (f_nok0 >= 0))[0]
mP = z26['mlg_s0_P'][:, -1, 0]
base_margin = mP[:128]
E_out = {
    'n_only_full': int(len(only_full)),
    'n_only_nok0': int(len(only_nok0)),
    'n_both': int(len(both)),
    'margin_med_only_nok0':
        float(np.median(
            base_margin[only_nok0]))
        if len(only_nok0) else None,
    'margin_med_only_full':
        float(np.median(
            base_margin[only_full]))
        if len(only_full) else None,
    'margin_med_both':
        float(np.median(
            base_margin[both]))
        if len(both) else None,
    'margin_med_rest':
        float(np.median(base_margin[
            np.where((f_full < 0)
                     & (f_nok0 < 0))[0]]))}
log('E k0 anatomy: only_nok0 margins %s '
    'vs rest %s (n=%d/%d)'
    % (E_out['margin_med_only_nok0'],
       E_out['margin_med_rest'],
       E_out['n_only_nok0'],
       128 - E_out['n_only_nok0']
       - E_out['n_both']))

# ================================================================
# verdict + result.json + npz
# ================================================================
tags = ['a_3137_ok',
        'i_sh_hard_ok' if ish_hard_ok
        else 'i_sh_hard_fail',
        'i_sh_soft85_ok' if ish_soft_ok
        else 'i_sh_soft85_below',
        'c_sh_hard_ok' if csh_hard_ok
        else 'c_sh_hard_fail',
        'c_sh_soft85_ok' if csh_soft_ok
        else 'c_sh_soft85_below',
        'probe_auc_gen_ok' if auc_gen_ok
        else 'probe_auc_gen_below',
        'k0_anatomy_recorded',
        'coverage_full']
verdict = '|'.join(tags)
runtime = time.time() - T0
log('VERDICT: %s' % verdict)
log('DONE (%.1fs)' % runtime)

result = {
    'phase': 3138,
    'name': NAME,
    'smoke': SMOKE,
    'verdict': verdict,
    'runtime_s': runtime,
    'seal_sha8': hashlib.sha256(
        json.dumps(SEAL, sort_keys=True,
                   ensure_ascii=False)
        .encode('utf-8')).hexdigest()[:8],
    'bank': {
        'bank_n': BANK_N,
        'n_templates': N_T,
        'dirs': list(DIRS),
        'shards': sorted(bank_ok.keys()),
        'lens_med': {
            '%s_T%d' % (dc, t):
                float(np.median(
                    lens_T[dc][t]))
            for dc in DIRS
            for t in range(N_T)}},
    'part_c': {
        'norm_share': {
            str(l): C_out['norm_share'][l]
            for l in ALL_L},
        'ish': {str(l): C_out['ish'][l]
                for l in ALL_L},
        'csh': {str(l): C_out['csh'][l]
                for l in ALL_L},
        'retr': {str(l): C_out['retr'][l]
                 for l in ALL_L},
        'resid_pca1_share': {
            str(l): C_out['resid'][l]
            for l in ALL_L},
        'ish_all_med': ish_all_med,
        'csh_all_med': csh_all_med,
        'gates': {
            'ish_hard': ish_hard_ok,
            'ish_soft85': ish_soft_ok,
            'csh_hard': csh_hard_ok,
            'csh_soft85': csh_soft_ok}},
    'part_d': {
        'auc': {str(l): D_out['auc'][l]
                for l in ALL_L},
        'auc_x': {str(l):
                  D_out['auc_x'][l]
                  for l in ALL_L},
        'auc_x_med': auc_x_med,
        'auc_gen_ok': auc_gen_ok},
    'part_e': E_out}
with io.open(os.path.join(
        OUT, 'result.json'), 'w',
        encoding='utf-8') as f:
    json.dump(result, f,
              ensure_ascii=False, indent=1)
np.savez(os.path.join(
    OUT, 'p136_readout.npz'),
    row_half=row_half,
    norm_share_B=np.array(
        [C_out['norm_share'][l]['B']
         for l in ALL_L]),
    norm_share_I=np.array(
        [C_out['norm_share'][l]['I']
         for l in ALL_L]),
    norm_share_C=np.array(
        [C_out['norm_share'][l]['C']
         for l in ALL_L]),
    norm_share_R=np.array(
        [C_out['norm_share'][l]['R']
         for l in ALL_L]),
    ish_med=np.array(
        [np.median(list(
            C_out['ish'][l].values()))
         for l in ALL_L]),
    csh_med=np.array(
        [np.median(list(
            C_out['csh'][l].values()))
         for l in ALL_L]),
    retr=np.array(
        [C_out['retr'][l]
         for l in ALL_L]),
    auc_raw_x=np.array(
        [D_out['auc_x'][l]['raw']
         for l in ALL_L]),
    auc_I_x=np.array(
        [D_out['auc_x'][l]['I']
         for l in ALL_L]),
    auc_BI_x=np.array(
        [D_out['auc_x'][l]['BI']
         for l in ALL_L]),
    auc_raw_T0=np.array(
        [D_out['auc'][l]['raw']
         for l in ALL_L]),
    auc_I_T0=np.array(
        [D_out['auc'][l]['I']
         for l in ALL_L]),
    fstep_w8full_36=f_full.astype(np.int8),
    fstep_w8_nok0_36=f_nok0.astype(np.int8))
log('dumps done: result.json + seal + '
    'p136_readout.npz + 8 bank shards')
if not SMOKE:
    if os.path.exists(CKPTF):
        os.remove(CKPTF)
        log('CKPT removed (data fully '
            'persisted)')
print('P3138 DONE (%.1fs)' % runtime)
