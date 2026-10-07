# -*- coding: utf-8 -*-
"""Phase 3107 (Omega-P105): Mode-family to write-head mapping.

T3 line, offline only (frozen captures from 3105/3106, no GPU
forward).  3105/3106 established that truth is tracked by the
model's own readout (m AUC 0.93-0.99) and by probes (AUC ~1.0),
but probes used ALL 2560 coordinates.  T3 asks WHERE the signal
lives:

  (1) SPARSITY: retrained probes restricted to Top-K |W|
      coordinates (K in {50,100,200,400} of 2560).  If a compact
      coordinate set ("write-head group" candidate) carries the
      signal, Top-100 AUC should approach full-dim AUC.
  (2) CROSS-MATERIAL REUSE: Top-200 coordinate sets from 3105
      (P/A1/A2 presence material) vs 3106 (chain/scatter
      material) - Jaccard overlap.  Same coordinates carrying
      truth across different materials = group reuse evidence.
  (3) READOUT ALIGNMENT: cosine between the zero-parameter
      direction w_dn = W_yes - W_no (lm_head rows) and learned
      probe weights W_T2; plus Spearman correlation between the
      two scores (m margin vs probe score) on test records.
  (4) LAYER STABILITY (descriptive): adjacent-layer Jaccard of
      Top-200 sets for crit_obj binding probe (T1) and truth
      probe (T2) across the 9 captured slots.

METHOD (pre-registered, frozen before computing any statistic):
  - Probe configs frozen: 3105 T2 at (last|L8, query_obj|L8,
    crit_obj|L3); 3106 T2 at (last|L8, query_obj|L8); 3105 T1
    8-way at (crit_obj|L0..L8) for layer stability.
  - lambda fixed to the value recorded in each phase's
    result.json (no re-selection -> no leakage).
  - Top-K coordinates selected by |W| (T2) or column L2 norm
    (T1) computed on TRAIN fit only; probe then REFIT in the
    Top-K subspace with the same lambda; evaluated on TEST.
  - Per-dim standardization by TRAIN stats (same as 3105/3106).
  - w_dn from safetensors (lm_head.weight, else tied
    model.embed_tokens.weight), YES_ID=9834 NO_ID=902.
  - Cosine in raw 2560-d coordinate space; chance |cos| for
    random 2560-d vectors ~ 1/sqrt(2560) ~ 0.02.

GATES (pre-registered):
  G1_sparse: for EACH frozen T2 config, Top-100 AUC >=
      0.90 * full-dim AUC on TEST.
  G2_reuse: Jaccard(Top200(3105 last|L8), Top200(3106
      last|L8)) >= 0.30.
  G3_align: |cos(w_dn, W_T2 last|L8 3105)| >= 0.15
      (~7x chance) AND Spearman(m, probe score) on 3105
      TEST >= 0.40.
  G4_layer_stability: descriptive only (curves registered).
  Verdict:
    G1&G2&G3 -> writehead_group_confirmed
    G1&!G2   -> sparse_but_material_specific
    !G1      -> distributed_signal
    G1&G2&!G3-> group_independent_of_readout_dir

SMOKE=1: subsample 300 records, slots {4,9}, K {50,200},
boot 300.  Output: tests/glm5/result/
rdc_query_construction_20260913/phase3107/
omega_p105_writehead_mapping/
"""
import gc
import hashlib
import io
import json
import os
import time
from datetime import datetime

import numpy as np

SMOKE = os.environ.get('SMOKE', '0') == '1'
ROOT = r'D:\AI2050\Ai2050-OpenOne'
MDIR = os.path.join(ROOT, 'models', 'hf', 'qwen3-4b')
R13 = (ROOT + r'\tests\glm5\result'
       r'\rdc_query_construction_20260913')
NAME = 'omega_p105_writehead_mapping'
OUT = os.path.join(R13, 'phase3107', NAME)
if SMOKE:
    OUT = os.path.join(OUT, 'smoke')
os.makedirs(OUT, exist_ok=True)
LOGF = os.path.join(OUT, 'run_log.txt')
T0 = time.time()

SEED = 31070
YES_ID, NO_ID = 9834, 902
LAMBDAS_FALLBACK = 0.1
K_LIST = [50, 200] if SMOKE else [50, 100, 200, 400]
TOPN = 200
N_SUB = 300 if SMOKE else None
N_BOOT = 300 if SMOKE else 2000

C5 = (R13 + r'\phase3105'
      r'\omega_p103_incontext_truth_consistency')
C6 = (R13 + r'\phase3106'
      r'\omega_p104_composition_dose_depth')
RES5 = C5 + r'\result.json'
RES6 = C6 + r'\result.json'


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


log('Phase 3107 Omega-P105 start; SMOKE=%d OUT=%s'
    % (SMOKE, OUT))

# ================================================================
# 0. Design seal (pre-computation)
# ================================================================
design = {
    'phase': 3107, 'name': NAME,
    'created': datetime.now().strftime(
        '%Y-%m-%d %H:%M:%S'),
    'smoke': SMOKE, 'seed': SEED,
    'probe_configs_frozen': {
        'T2_3105': ['last|L8', 'query_obj|L8',
                    'crit_obj|L3'],
        'T2_3106': ['last|L8', 'query_obj|L8'],
        'T1_3105_layer_sweep': ['crit_obj|L0',
                                'crit_obj|L1',
                                'crit_obj|L2',
                                'crit_obj|L3',
                                'crit_obj|L4',
                                'crit_obj|L5',
                                'crit_obj|L6',
                                'crit_obj|L7',
                                'crit_obj|L8']},
    'k_list': K_LIST, 'topn': TOPN,
    'lambda_source': 'recorded lam in each phase '
                     'result.json (no re-selection)',
    'topk_protocol': 'select by |W| (T2) / column L2 '
                     'norm (T1) on TRAIN fit, refit '
                     'same-lambda in subspace, eval '
                     'TEST',
    'gates': {
        'G1_sparse': 'Top-100 AUC >= 0.90 x full-dim '
                     'AUC per frozen T2 config (TEST)',
        'G2_reuse': 'Jaccard Top200(3105 last|L8, '
                    '3106 last|L8) >= 0.30',
        'G3_align': '|cos(w_dn, W_T2 3105 last|L8)| '
                    '>= 0.15 AND Spearman(m, probe) '
                    'on 3105 TEST >= 0.40',
        'G4_layer_stability': 'descriptive curves'},
    'verdict_map': {
        'G1&G2&G3': 'writehead_group_confirmed',
        'G1&!G2': 'sparse_but_material_specific',
        '!G1': 'distributed_signal',
        'G1&G2&!G3': 'group_independent_of_'
                     'readout_dir'},
}
with io.open(os.path.join(OUT, 'design_seal.json'),
             'w', encoding='utf-8') as f:
    json.dump(design, f, indent=1)
log('design sealed (pre-computation)')

# ================================================================
# 1. w_dn from safetensors (no full model load; bf16 needs
#    framework='pt' since numpy has no bfloat16)
# ================================================================
from safetensors import safe_open  # noqa: E402

w_dn = None
src_name = None
for fn in sorted(os.listdir(MDIR)):
    if not fn.endswith('.safetensors'):
        continue
    try:
        with safe_open(os.path.join(MDIR, fn),
                       framework='pt') as f:
            keys = list(f.keys())
            for cand in ('lm_head.weight',
                         'model.embed_tokens.weight'):
                if cand in keys:
                    W = f.get_tensor(cand)
                    W = W[[YES_ID, NO_ID]] \
                        .float().cpu().numpy()
                    w_dn = W[0].astype(np.float32) \
                        - W[1].astype(np.float32)
                    src_name = '%s:%s' % (fn, cand)
                    break
    except Exception as e:
        log('skip %s: %r' % (fn, e))
    if w_dn is not None:
        break
assert w_dn is not None, 'lm_head row not found'
log('w_dn loaded from %s; |w|=%.3f norm=%.3f'
    % (src_name, float(np.abs(w_dn).sum()),
       float(np.linalg.norm(w_dn))))

# ================================================================
# 2. Load frozen captures
# ================================================================
def load_cap(cdir):
    z = np.load(os.path.join(cdir, 'capture.npz'),
                allow_pickle=False)
    keys = ('split', 'cond', 'truth', 'crit_rel',
            'query_rel', 'tag')
    meta = {k: z[k] for k in keys if k in z}
    if 'family' in z:
        meta['family'] = z['family']
    return z, meta


z5, m5 = load_cap(C5)
z6, m6 = load_cap(C6)
log('captures loaded: 3105 X=%s, 3106 X=%s'
    % (str(z5['X'].shape), str(z6['X'].shape)))

res5 = json.load(io.open(RES5, encoding='utf-8'))
res6 = json.load(io.open(RES6, encoding='utf-8'))


def lam_of(res, name):
    v = res['T2_ctx'].get(name) or \
        res['T1_ctx'].get(name)
    assert v is not None, name
    return float(v['lam'])


# ================================================================
# 3. Probe machinery
# ================================================================
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


def spearman(a, b):
    ra = np.argsort(np.argsort(a)).astype(np.float64)
    rb = np.argsort(np.argsort(b)).astype(np.float64)
    ra = ra - ra.mean()
    rb = rb - rb.mean()
    d = np.sqrt((ra * ra).sum() * (rb * rb).sum())
    if d == 0:
        return float('nan')
    return float((ra * rb).sum() / d)


def get_slice(z, pos_name, li):
    POSITIONS = ['pos0', 'crit_pred', 'crit_obj',
                 'query_pred', 'query_obj', 'last']
    p = POSITIONS.index(pos_name)
    X = z['X'][:, p, li, :].astype(np.float32)
    return X


def t2_eval(z, meta, pos_name, li, lam, phase_res,
            k_list=()):
    """Full-dim + Top-K refit T2 truth probe.
    Returns dict with full auc and per-K auc + coords."""
    X = get_slice(z, pos_name, li)
    y = meta['truth'].astype(np.int64)
    tag = meta['tag']
    # main records only
    if 'family' in meta:
        keep = np.ones(len(y), dtype=bool)
    else:
        keep = (tag == b'main') | (tag == 'main')
    if N_SUB:
        rs = np.random.RandomState(SEED)
        idx_all = np.where(keep)[0]
        sub = rs.choice(idx_all, min(N_SUB, len(idx_all)),
                        replace=False)
        keep = np.zeros(len(y), dtype=bool)
        keep[sub] = True
    X = X[keep]
    y = y[keep]
    sp = meta['split'][keep]
    tr = np.where(sp == b'train')[0] \
        if sp.dtype.kind == 'S' \
        else np.where(sp == 'train')[0]
    te = np.where(sp == b'test')[0] \
        if sp.dtype.kind == 'S' \
        else np.where(sp == 'test')[0]
    teE = np.where(sp == b'testE')[0] \
        if sp.dtype.kind == 'S' \
        else np.where(sp == 'testE')[0]
    assert len(tr) > 0, ('empty train', pos_name, li)
    assert len(te) > 0, ('empty test', pos_name, li)
    assert len(set(y[tr].tolist())) == 2, \
        ('train single class', pos_name, li)
    mu = X[tr].mean(0)
    sd = X[tr].std(0) + 1e-6
    Z = (X - mu) / sd
    Yv = (y * 2.0 - 1.0).astype(np.float32).reshape(-1, 1)
    W = ridge_solve(Z[tr], Yv[tr], lam)
    out = {'lam': lam, 'pos': pos_name, 'L': li,
           'full_auc_test': auc_score(y[te],
                                      (Z[te] @ W)[:, 0]),
           'full_auc_testE': auc_score(y[teE],
                                       (Z[teE] @ W)[:, 0]),
           'w': W[:, 0], 'mu': mu, 'sd': sd,
           'Zte': Z[te], 'yte': y[te],
           'te_idx': np.where(keep)[0][te],
           'k_aucs': {}, 'k_coords': {}}
    order = np.argsort(-np.abs(W[:, 0]))
    for K in k_list:
        sel = np.sort(order[:K])
        Wk = ridge_solve(Z[tr][:, sel], Yv[tr], lam)
        out['k_aucs'][K] = {
            'test': auc_score(y[te],
                              (Z[te][:, sel] @ Wk)[:, 0]),
            'testE': auc_score(y[teE],
                               (Z[teE][:, sel] @ Wk)[:, 0])}
        out['k_coords'][K] = sel
    del X, Z
    gc.collect()
    return out


def top_coords(W, k):
    if W.ndim == 1:
        sc = np.abs(W)
    else:
        sc = np.linalg.norm(W, axis=1)
    return np.sort(np.argsort(-sc)[:k])


def jaccard(a, b):
    sa, sb = set(a.tolist()), set(b.tolist())
    return len(sa & sb) / max(1, len(sa | sb))


# ================================================================
# 4. G1 sparsity + per-config results
# ================================================================
g1_items = []
t2_results = {}

def _fmt(v):
    return '%.4f' % v if v is not None else 'n/a'


cfgs_5 = [('last', 8), ('query_obj', 8), ('crit_obj', 3)]
cfgs_6 = [('last', 8), ('query_obj', 8)]
for (pn, li) in cfgs_5:
    nm = '%s|L%d' % (pn, li)
    r = t2_eval(z5, m5, pn, li, lam_of(res5, nm),
                res5, K_LIST)
    t2_results['3105:%s' % nm] = r
    ok1 = None
    if 100 in r['k_aucs']:
        f = r['full_auc_test']
        k1 = r['k_aucs'][100]['test']
        ok1 = (k1 is not None and f is not None
               and k1 >= 0.90 * f)
    g1_items.append(('3105:%s' % nm, ok1, r))
    log('G1 3105 %s: full=%s k100=%s ok=%s'
        % (nm, _fmt(r['full_auc_test']),
           _fmt(r['k_aucs'].get(100, {}).get('test')),
           ok1))
for (pn, li) in cfgs_6:
    nm = '%s|L%d' % (pn, li)
    r = t2_eval(z6, m6, pn, li, lam_of(res6, nm),
                res6, K_LIST)
    t2_results['3106:%s' % nm] = r
    ok1 = None
    if 100 in r['k_aucs']:
        f = r['full_auc_test']
        k1 = r['k_aucs'][100]['test']
        ok1 = (k1 is not None and f is not None
               and k1 >= 0.90 * f)
    g1_items.append(('3106:%s' % nm, ok1, r))
    log('G1 3106 %s: full=%s k100=%s ok=%s'
        % (nm, _fmt(r['full_auc_test']),
           _fmt(r['k_aucs'].get(100, {}).get('test')),
           ok1))
G1 = all(ok is True for (_, ok, _) in g1_items)
log('G1 sparse: %s' % G1)

# ================================================================
# 5. G2 cross-material reuse
# ================================================================
w5 = t2_results['3105:last|L8']['w']
w6 = t2_results['3106:last|L8']['w']
top5 = top_coords(w5, TOPN)
top6 = top_coords(w6, TOPN)
jac_5_6 = jaccard(top5, top6)
G2 = jac_5_6 >= 0.30
log('G2 reuse: Jaccard(3105,3106 Top200 last|L8) = '
    '%.4f -> %s' % (jac_5_6, G2))

# auxiliary overlaps
aux = {}
aux['jaccard_3105_query_obj_vs_last'] = jaccard(
    top_coords(t2_results['3105:query_obj|L8']['w'],
               TOPN), top5)
# 3106 internal: chain-vs-scatter refit (last|L8)
fam6 = z6['family']
tr6 = np.where((m6['split'] == b'train'))[0] \
    if m6['split'].dtype.kind == 'S' \
    else np.where((m6['split'] == 'train'))[0]
X6 = get_slice(z6, 'last', 8)
mu6 = X6[tr6].mean(0)
sd6 = X6[tr6].std(0) + 1e-6
Z6 = (X6 - mu6) / sd6
y6 = m6['truth'].astype(np.int64)
lam6 = lam_of(res6, 'last|L8')
w_ch = w_sc = None
for fam_val, nm_f in ((b'chain', 'chain'),
                      ('chain', 'chain')):
    pass
fam_arr = [x.decode() if isinstance(x, bytes) else x
           for x in fam6]
sp_arr = [x.decode() if isinstance(x, bytes) else x
          for x in m6['split']]
trm = np.array([s == 'train' for s in sp_arr])
Yf = ((y6 * 2 - 1).astype(np.float32)
      .reshape(-1, 1))
for fam_val in ('chain', 'scatter'):
    fm = np.array([f == fam_val for f in fam_arr])
    sel = trm & fm
    Wf = ridge_solve(Z6[sel], Yf[sel], lam6)
    if fam_val == 'chain':
        w_ch = Wf[:, 0]
    else:
        w_sc = Wf[:, 0]
aux['jaccard_3106_chain_vs_scatter'] = jaccard(
    top_coords(w_ch, TOPN), top_coords(w_sc, TOPN))
aux['jaccard_3105_vs_3106_query_obj'] = jaccard(
    top_coords(t2_results['3105:query_obj|L8']['w'],
               TOPN),
    top_coords(t2_results['3106:query_obj|L8']['w'],
               TOPN))
log('aux overlaps: %s' % json.dumps(
    {k: round(v, 4) for k, v in aux.items()}))
del X6, Z6
gc.collect()

# ================================================================
# 6. G3 readout alignment
# ================================================================
w5n = w5 / (np.linalg.norm(w5) + 1e-12)
wdnn = w_dn / (np.linalg.norm(w_dn) + 1e-12)
cos_5 = float(np.abs(np.dot(w5n, wdnn)))
w6n = w6 / (np.linalg.norm(w6) + 1e-12)
cos_6 = float(np.abs(np.dot(w6n, wdnn)))
# Spearman between m margin and probe score on 3105 TEST
r5t2 = t2_results['3105:last|L8']
Zte = r5t2['Zte']
probe_score = (Zte @ r5t2['w'])
if probe_score.ndim > 1:
    probe_score = probe_score[:, 0]
m_all5 = z5['m'][r5t2['te_idx']]
rho_m_probe = spearman(m_all5, probe_score)
G3 = (cos_5 >= 0.15) and (rho_m_probe >= 0.40)
log('G3 align: |cos(w_dn,W5 last|L8)|=%.4f '
    '(3106 %.4f); Spearman(m,probe)=%.4f -> %s'
    % (cos_5, cos_6, rho_m_probe, G3))

# per-layer cos curve (3105 last position)
cos_layers = {}
for li in range(9):
    X = get_slice(z5, 'last', li)
    sp = [x.decode() if isinstance(x, bytes) else x
          for x in m5['split']]
    tag5 = [x.decode() if isinstance(x, bytes) else x
            for x in m5['tag']]
    trm = np.array([s == 'train' and t == 'main'
                    for s, t in zip(sp, tag5)])
    mu = X[trm].mean(0)
    sd = X[trm].std(0) + 1e-6
    Z = (X - mu) / sd
    y5 = m5['truth'].astype(np.int64)
    Yv = (y5 * 2 - 1).astype(np.float32) \
        .reshape(-1, 1)
    W = ridge_solve(Z[trm], Yv[trm], lam_of(
        res5, 'last|L%d' % li))
    wn = W[:, 0] / (np.linalg.norm(W[:, 0]) + 1e-12)
    cos_layers['L%d' % li] = float(
        np.abs(np.dot(wn, wdnn)))
    del X, Z
    gc.collect()
log('cos(w_dn, W_T2 last) per layer: %s'
    % json.dumps({k: round(v, 4)
                  for k, v in cos_layers.items()}))

# ================================================================
# 7. G4 layer stability (descriptive)
# ================================================================
stab = {'T1_crit_obj_adjacent_jaccard': [],
        'T2_last_adjacent_jaccard': []}
sp5 = [x.decode() if isinstance(x, bytes) else x
       for x in m5['split']]
tag5 = [x.decode() if isinstance(x, bytes) else x
        for x in m5['tag']]
trm5 = np.array([s == 'train' for s in sp5])
y5 = m5['truth'].astype(np.int64)
prev_t1 = prev_t2 = None
for li in range(9):
    # T1: 8-way crit_rel at crit_obj
    X = get_slice(z5, 'crit_obj', li)
    mu = X[trm5].mean(0)
    sd = X[trm5].std(0) + 1e-6
    Z = (X - mu) / sd
    cr = m5['crit_rel'].astype(np.int64)
    Y1 = np.eye(8, dtype=np.float32)[cr]
    lam1 = lam_of(res5, 'crit_obj|L%d' % li)
    W1 = ridge_solve(Z[trm5], Y1[trm5], lam1)
    c1 = top_coords(W1, TOPN)
    # T2 at last
    X2 = get_slice(z5, 'last', li)
    mu2 = X2[trm5].mean(0)
    sd2 = X2[trm5].std(0) + 1e-6
    Z2 = (X2 - mu2) / sd2
    Yv2 = (y5 * 2 - 1).astype(np.float32) \
        .reshape(-1, 1)
    W2 = ridge_solve(Z2[trm5], Yv2[trm5], lam_of(
        res5, 'last|L%d' % li))
    c2 = top_coords(W2, TOPN)
    if prev_t1 is not None:
        stab['T1_crit_obj_adjacent_jaccard'].append(
            jaccard(prev_t1, c1))
        stab['T2_last_adjacent_jaccard'].append(
            jaccard(prev_t2, c2))
    prev_t1, prev_t2 = c1, c2
    del X, Z, X2, Z2
    gc.collect()
log('G4 stability curves: %s'
    % json.dumps({k: [round(x, 3) for x in v]
                  for k, v in stab.items()}))

# ================================================================
# 8. Verdict & save
# ================================================================
n_pass = sum([G1, G2, G3])
if G1 and G2 and G3:
    verdict = 'writehead_group_confirmed'
elif not G1:
    verdict = 'distributed_signal'
elif not G2:
    verdict = 'sparse_but_material_specific'
else:
    verdict = 'group_independent_of_readout_dir'
log('GATES: G1=%s G2=%s G3=%s -> %s'
    % (G1, G2, G3, verdict))

def _strip(r):
    out = {}
    for k, v in r.items():
        if k in ('w', 'mu', 'sd', 'Zte', 'yte',
                 'te_idx'):
            continue
        if k == 'k_coords':
            out[k] = {str(kk): vv.tolist()
                      for kk, vv in v.items()}
        elif isinstance(v, np.generic):
            out[k] = v.item()
        elif isinstance(v, np.ndarray):
            out[k] = v.tolist()
        else:
            out[k] = v
    return out


results = {
    'verdict': verdict,
    'gates': {
        'G1_sparse': G1,
        'G2_reuse': G2,
        'G3_align': G3,
        'g1_items': [(nm, ok) for (nm, ok, _)
                     in g1_items],
        'jaccard_3105_3106_lastL8': jac_5_6,
        'cos_wdn_W5_lastL8': cos_5,
        'cos_wdn_W6_lastL8': cos_6,
        'spearman_m_probe_3105_test': rho_m_probe},
    't2_configs': {'3105:%s' % nm: _strip(r)
                   for nm, r in
                   [('last|L8', t2_results['3105:last|L8']),
                    ('query_obj|L8',
                     t2_results['3105:query_obj|L8']),
                    ('crit_obj|L3',
                     t2_results['3105:crit_obj|L3'])]},
    'aux_overlaps': aux,
    'cos_layers_3105_last': cos_layers,
    'layer_stability': stab,
    'w_dn_source': src_name,
    'created': datetime.now().strftime(
        '%Y-%m-%d %H:%M:%S'),
}
with io.open(os.path.join(OUT, 'result.json'), 'w',
             encoding='utf-8') as f:
    json.dump(results, f, indent=1)
log('result.json written; verdict=%s' % verdict)
print('PHASE3107_DONE verdict=%s G1=%s G2=%s G3=%s'
      % (verdict, G1, G2, G3))
