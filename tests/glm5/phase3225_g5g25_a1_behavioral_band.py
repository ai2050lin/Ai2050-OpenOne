# -*- coding: utf-8 -*-
"""Phase 3225 (G5-G25 A1): pre-softmax behavioral band — compute + seal arm.

Per prereg in MEMO Phase 3224 tail (default candidate (a); plan A weights
acquisition authorized via user's continuation instruction "好的，继续"
interpreted as the preregistered-default-arm trigger — honest record).

Design doc: phase3223 presoftmax_band_design_v1.json (87e51bcd). This arm:
  STEP 1 (FREEZE, before any displacement observation): pin scale alpha in
  {0.1, 0.5} (0.5 = gate scale / nonlinearity stress, 0.1 = linearity
  control), S rule (top-64 tokens by row-mean |dl_true| + gold class
  tokens), row mapping (r -> template t=r//730, pos r%730), injection layer
  = FINAL cached layer NH-1 (directions u*/v_hat live at readout kout=NH-2;
  injecting kout-layer directions at the final layer IS the test — logged),
  features P1=u* (FTR-26, 3194 protocol) / P2=v_hat_held (FTR-25, 3191
  protocol), 4 OOV-class folds (6..9), models qwen3-4b + qwen3-14b;
  FTR-15 deferred (no single direction; entity-LOO ratio structure needs
  its own mapping). Freeze sha written before STEP 2.
  STEP 2 (COMPUTE): device anchors first — Qn pipeline rebuilt verbatim from
  sealed 3194 recipe (module import), drift vs sealed 3169/3189/3190/3191
  scalars == 0 (1e-9); then exact/linearized logit displacements:
    dl_true(t) = W_U @ (lnf(h+dv) - lnf(h)),  lnf(x) = x/rms(x) (*) gamma
    dl_lin(t)  = W_U @ J @ dv,  J = (diag(gamma) - (gamma*xhat) xhat^T)/rms
  with dv = unit(P) * alpha * rms(h_row) per held row.
  Weights: plan A shards only (4B shard1 embed(tied W_U) + shard3 norm;
  14B shard8 lm_head + norm). CPU matrix math only — no model forward,
  no GPU.
  STEP 3 (SEAL): per-model npz (fold x predictor x alpha metrics + vectors)
  + result.json. Gate evaluation deferred to 3226 (A2) per design chain.
"""
import datetime
import glob
import hashlib
import json
import os
import sys
import time

import numpy as np

T0 = time.time()
try:
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
except Exception:
    pass
sys.path.insert(0, r'D:\AI2050\Ai2050-OpenOne\tests\glm5')
import phase3194_g5d2_ftr26_heldout_upgrade as p94  # sealed recipe reuse

ROOT = p94.ROOT
SRC_DIR = p94.SRC_DIR
NAME = 'g5g25_a1_behavioral_band_compute'
OUTDIR = os.path.join(SRC_DIR, 'phase3225', NAME)
SMOKE = os.environ.get('P3225_SMOKE') == '1'
PFX = 'smoke_' if SMOKE else ''
HUB = os.path.join(os.path.expanduser('~'), '.cache', 'huggingface', 'hub')
LEDGER = os.path.join(ROOT, 'research', 'gpt5', 'atlas', 'atlas_ledger.json')


def sha8_file(p):
    return hashlib.sha256(open(p, 'rb').read()).hexdigest()[:8]


LOG = []


def log(s):
    LOG.append(s)
    print(s, flush=True)


os.makedirs(OUTDIR, exist_ok=True)

MODELS_A1 = ['qwen3-4b', 'qwen3-14b']
ALPHAS = (0.1, 0.5)
GATE_ALPHA = 0.5
S_TOPK = 64
CLS_IDS = [6, 7, 8, 9]
CLS_NAMES = ['乐器', '天气', '运动', '电器']
GOLD_TOKENS = {'qwen3-4b': [112946, 104307, 101079, 105724],
               'qwen3-14b': [112946, 104307, 101079, 105724]}
WEIGHT_FILES = {
    'qwen3-4b': [('model-00001-of-00003.safetensors', ['model.embed_tokens.weight']),
                 ('model-00003-of-00003.safetensors', ['model.norm.weight'])],
    'qwen3-14b': [('model-00008-of-00008.safetensors', ['lm_head.weight', 'model.norm.weight'])],
}
SHARD_SHA8 = {'qwen3-4b': {'model-00001-of-00003.safetensors': '4c807e25',
                           'model-00003-of-00003.safetensors': 'c7b1aa8f'},
              'qwen3-14b': {'model-00008-of-00008.safetensors': '0d6b9229'}}

# ---------- STEP 1: FREEZE (before any displacement observation) ----------
freeze = {
    'phase': 3225, 'name': NAME,
    'prereg': 'MEMO Phase 3224 tail (a); design 87e51bcd; user authorization '
              'record: "好的，继续" = preregistered-default-arm trigger, plan A',
    'scale_pinned': {'alphas': list(ALPHAS), 'gate_alpha': GATE_ALPHA,
                     'dv_norm': 'unit(P) * alpha * rms(h_row) per row'},
    's_rule': 'top-%d vocab ids by row-mean |dl_true| (per fold x P x alpha), '
              'plus gold class tokens (dedup)' % S_TOPK,
    'row_mapping': 'row r of Y (3169 B-protocol 876 rows) -> template '
                   't=r//730, pos r%730; h_final = H16[t,pos,NH-1,:] fp32',
    'injection_layer': 'FINAL cached layer NH-1; directions u*/v_hat live at '
                       'readout kout=NH-2 — injecting kout directions at the '
                       'final layer is exactly the behavioral test',
    'features': {'P1': 'u* (FTR-26, 3194 protocol verbatim)',
                 'P2': 'v_hat_held (FTR-25, 3191 protocol verbatim)'},
    'folds': '4 OOV classes %s (ids %s)' % (CLS_NAMES, CLS_IDS),
    'gold_tokens': GOLD_TOKENS,
    'models': MODELS_A1,
    'ftr15_deferred': 'no single direction (entity-LOO ratio structure); '
                      'behavioral mapping needs separate design append',
    'weight_anchors': SHARD_SHA8,
    'device_anchor_protocol': 'Qn rebuilt verbatim (3194 recipe import); '
                              'drift vs sealed 3169 E per-seed (1e-9) and '
                              '3189/3190/3191 scalars (== 0)',
}
fz_blob = json.dumps(freeze, ensure_ascii=False, indent=1, sort_keys=True)
DESIGN_SHA = hashlib.sha256(fz_blob.encode('utf-8')).hexdigest()[:8]
freeze['design_sha8'] = DESIGN_SHA
with open(os.path.join(OUTDIR, PFX + 'execution.json'), 'wb') as f:
    f.write(json.dumps(freeze, ensure_ascii=False, indent=1).encode('utf-8'))
log('FREEZE sealed: design_sha8=%s (before any displacement observation)'
    % DESIGN_SHA)

# ---------- STEP 2: COMPUTE ----------
from safetensors import safe_open
from tokenizers import Tokenizer


def lnf(x, g, eps):
    return (x / np.sqrt((x ** 2).mean(axis=-1, keepdims=True) + eps)) * g


def find_snapshot(repo_dirname):
    return glob.glob(os.path.join(HUB, repo_dirname, 'snapshots', '*'))[0]


SNAP = {'qwen3-4b': find_snapshot('models--Qwen--Qwen3-4B-Base'),
        'qwen3-14b': find_snapshot('models--Qwen--Qwen3-14B')}
TK = {}
for m in MODELS_A1:
    tp = glob.glob(os.path.join(SNAP[m], 'tokenizer.json'))[0]
    TK[m] = Tokenizer.from_file(tp)
    gold = [TK[m].encode(c).ids[0] for c in CLS_NAMES]
    assert gold == GOLD_TOKENS[m], (m, gold)
log('gold tokens verified from disk tokenizers')

sealed_3169 = json.load(open(p94.SRC['p3169_result'], encoding='utf-8'))
sealed_3189 = json.load(open(p94.SRC['p3189_result'], encoding='utf-8'))
sealed_3190 = json.load(open(p94.SRC['p3190_result'], encoding='utf-8'))
sealed_3191 = json.load(open(p94.SRC['p3191_result'], encoding='utf-8'))

CFG = {}
for m in MODELS_A1:
    CFG[m] = json.load(open(os.path.join(SNAP[m], 'config.json'),
                            encoding='utf-8'))
EPS = {m: float(CFG[m].get('rms_norm_eps', 1e-6)) for m in MODELS_A1}

out = {}
npz_payload = {}
for m in MODELS_A1:
    log('=== %s ===' % m)
    rd = int(sealed_3169['per_model'][m]['readout'])
    z = np.load(p94.SRC[p94.NPZ69_KEY[m]])
    H16 = z['H']
    NTz, NPz, NHz, Dz = H16.shape
    kout = NHz - 2
    assert kout == rd, ('readout slot', m)
    NT, NP_ = p94.NT, p94.NP_
    assert (NTz, NPz) == (NT, NP_)
    Y = H16[:, :, kout, :].reshape(NT * NP_, Dz).astype(np.float32)
    H_final = H16[:, :, NHz - 1, :].reshape(NT * NP_, Dz).astype(np.float32)

    # --- Qn pipeline verbatim (3194 run_arm lines 300-338) ---
    tr_base = {}
    for s in p94.SEEDS_S1:
        tr_pairs, te_pairs = p94.split_s1_246(s)
        tr_base[s] = [t * NP_ + p94.PI_OF_PAIR[p] for t in range(NT)
                      for p in p94.PAIRS if p in tr_pairs]
    oov_rows = p94.ALL_OOV_ROWS
    oov_row_cls = np.array([p94.PAIRS[r % NP_][1] for r in oov_rows])

    def rv(r, ne):
        i, c = p94.PAIRS[r % NP_]
        return p94.phi_row(i, c, r // NP_, ne, p94.NC, p94.NT)

    E69 = sealed_3169['per_model'][m]['B']
    E89 = sealed_3189['per_model'][m]
    drift69 = 0.0
    Q_rows = np.zeros((len(oov_rows), Dz), dtype=np.float32)
    for si, s in enumerate(p94.SEEDS_S1):
        tr_rows = tr_base[s]
        Xtr = np.stack([rv(r, p94.NE) for r in tr_rows])
        W = p94.ridge_primal(Xtr, Y[tr_rows])
        ref = Y[tr_rows].mean(0)
        Dk = float(((Y[tr_rows] - ref) ** 2).sum(1).mean()) + 1e-9
        te_pairs = p94.split_s1_246(s)[1]
        te_rows = [t * NP_ + p94.PI_OF_PAIR[p] for t in range(NT)
                   for p in p94.PAIRS if p in te_pairs]
        Xs = np.stack([rv(r, p94.NE) for r in te_rows])
        es = ((Xs @ W - Y[te_rows]) ** 2).sum(1) / Dk
        Xo = np.stack([rv(r, p94.NE) for r in oov_rows])
        eo = ((Xo @ W - Y[oov_rows]) ** 2).sum(1) / Dk
        drift69 = max(drift69, abs(float(es.mean()) - E69['E_seen_per_seed'][si]),
                      abs(float(eo.mean()) - E69['E_oov_per_seed'][si]))
        R_oov = Y[oov_rows] - Xo @ W
        for j in range(len(oov_rows)):
            rvv = R_oov[j]
            Q_rows[j] += rvv / (float(np.linalg.norm(rvv)) + 1e-18)
        del W, Xtr
    assert drift69 < 1e-9, ('3169 per-seed E replication', m, drift69)
    Qn = Q_rows / len(p94.SEEDS_S1)
    Qn = Qn / (np.linalg.norm(Qn, axis=1, keepdims=True) + 1e-18)

    # --- device anchors: 3189/3190/3191 scalars ---
    s0 = np.linalg.svd(Qn, compute_uv=False)
    energy = s0 ** 2
    tot = float(energy.sum())
    top1 = float(energy[0] / tot)
    cls_ids = CLS_IDS
    V = []
    within = []
    for c in cls_ids:
        idxs = np.where(oov_row_cls == c)[0]
        V.append(p94.unit_mean(Qn[idxs]))
        Cc = Qn[idxs] @ Qn[idxs].T
        within.append(float(Cc[np.triu_indices(len(idxs), 1)].mean()))
    V = np.stack(V)
    Ccross = V @ V.T
    cross_med = float(np.median([float(Ccross[i, j])
                                 for i in range(4) for j in range(i + 1, 4)]))
    drift89 = max(abs(top1 - E89['top1']),
                  max(abs(a - b) for a, b in zip(within, E89['within_class'])),
                  abs(cross_med - E89['cross_med']))
    assert drift89 == 0.0, ('3189 scalar drift', m, drift89)
    U, S, Vt = np.linalg.svd(Qn, full_matrices=False)
    ustar = Vt[0]
    loo = []
    for c in cls_ids:
        keep = oov_row_cls != c
        sl = np.linalg.svd(Qn[keep], compute_uv=False)
        el = sl ** 2
        loo.append(float(el[0] / el.sum()))
    loo_max = float(max(abs(x - top1) for x in loo))
    med_abs = []
    for c in cls_ids:
        held = oov_row_cls == c
        v_hat = p94.unit_mean(Qn[~held])
        cos_held = Qn[held] @ v_hat
        med_abs.append(float(np.median(np.abs(cos_held))))
    E90, E91 = sealed_3190['per_model'][m], sealed_3191['per_model'][m]
    drift90 = max(abs(loo_max - E90['loo_max_delta']),
                  abs(top1 - E90['top1']), abs(cross_med - E90['cross_med']),
                  max(abs(a - b) for a, b in zip(loo, E90['loo_top1'])))
    assert drift90 == 0.0, ('3190 scalar drift', m, drift90)
    drift91 = max(abs(float(min(med_abs)) - E91['min_med_abs_cos']),
                  max(abs(a - b) for a, b in zip(med_abs,
                                                 E91['med_abs_cos_held'])))
    assert drift91 == 0.0, ('3191 control-anchor drift', m, drift91)
    log('  device anchors OK: drift69=%.2e drift89=0 drift90=0 drift91=0'
        % drift69)

    # --- load plan A weights ---
    tensors = {}
    for fn, keys in WEIGHT_FILES[m]:
        sp = os.path.join(SNAP[m], fn)
        assert sha8_file(sp) == SHARD_SHA8[m][fn], ('shard sha', m, fn)
        with safe_open(sp, framework='pt') as f:
            avail = list(f.keys())
            for k in keys:
                assert k in avail, ('key missing', m, fn, k, avail[:8])
                tensors[k] = f.get_tensor(k).float().numpy()
    wu_key = 'model.embed_tokens.weight' if 'model.embed_tokens.weight' in tensors \
        else 'lm_head.weight'
    W_U = tensors[wu_key]
    gamma = tensors['model.norm.weight'].reshape(-1)
    assert W_U.shape[0] == CFG[m]['vocab_size'] and W_U.shape[1] == Dz
    log('  weights: %s %s %s | norm %s | eps=%g' % (
        wu_key, W_U.shape, W_U.dtype, gamma.shape, EPS[m]))

    # --- behavioral displacement per fold x predictor x alpha ---
    mm = {}
    for ci, c in enumerate(cls_ids):
        held_rows = np.array([oov_rows[j] for j in np.where(oov_row_cls == c)[0]])
        h = H_final[held_rows]
        rms = np.sqrt((h ** 2).mean(axis=1, keepdims=True) + EPS[m])
        xhat = h / rms
        # J self-check (once per model): numeric d(lnf)/dh vs closed form
        # (1/rms)(diag(gamma) - (gamma*xhat)xhat^T/D) — validates 1/N factor
        if 'jchk_done' not in dir():
            jchk = {}
        if m not in jchk:
            jchk[m] = True
            h0 = h[:1]
            r0 = np.sqrt((h0 ** 2).mean(axis=1, keepdims=True) + EPS[m])
            xh0 = h0 / r0
            dv0 = ustar[None, :] * (0.01 * r0)
            num = (lnf(h0 + dv0, gamma, EPS[m]) - lnf(h0 - dv0, gamma, EPS[m]))
            num = num / (2 * 0.01 * r0)
            ana = (gamma[None, :] * dv0 / r0
                   - (gamma[None, :] * xh0) * (xh0 * dv0).sum(1, keepdims=True)
                   / (Dz * r0))
            cnum = float((num * ana).sum(1).mean()
                         / (np.linalg.norm(num) * np.linalg.norm(ana) + 1e-18))
            assert cnum > 0.9999, ('J self-check', m, cnum)
            log('  J self-check numeric-vs-analytic cos=%.8f' % cnum)

        gold = GOLD_TOKENS[m][ci]
        for pname, dv_unit in (('P1_ustar', ustar), ('P2_vhat', p94.unit_mean(Qn[oov_row_cls != c]))):
            for a in ALPHAS:
                dv = dv_unit[None, :] * (a * rms)
                d_true = lnf(h + dv, gamma, EPS[m]) - lnf(h, gamma, EPS[m])
                # linearized: J @ dv = (gamma*dv - (gamma*xhat)(xhat.dv)) / rms
                gprod = gamma[None, :] * dv
                corr = (gamma[None, :] * xhat) * (xhat * dv).sum(
                    1, keepdims=True) / Dz
                jdv = (gprod - corr) / rms
                LT = d_true @ W_U.T          # (R, V) exact
                LP = jdv @ W_U.T             # (R, V) linearized
                nt_ = np.linalg.norm(LT, axis=1)
                np_ = np.linalg.norm(LP, axis=1)
                cos_full = (LP * LT).sum(1) / (np_ * nt_ + 1e-18)
                band = np.abs(np_ - nt_) / (nt_ + 1e-18)
                smean = np.abs(LT).mean(0)
                Sidx = np.argsort(-smean)[:S_TOPK]
                Sidx = np.unique(np.concatenate([Sidx, [gold]]))
                lts, lps = LT[:, Sidx], LP[:, Sidx]
                cos_S = (lps * lts).sum(1) / (
                    np.linalg.norm(lps, axis=1) * np.linalg.norm(lts, axis=1) + 1e-18)
                el = np.linalg.svd(lts, compute_uv=False) ** 2
                svd8 = float(el[:8].sum() / el.sum())
                key = 'f%d_%s_a%s' % (c, pname, a)
                mm[key] = dict(
                    cos_full_mean=float(cos_full.mean()),
                    cos_full_med=float(np.median(cos_full)),
                    cos_S_mean=float(cos_S.mean()),
                    cos_S_med=float(np.median(cos_S)),
                    band_med=float(np.median(band)),
                    band_mean=float(band.mean()),
                    svd8_energy=svd8,
                    gold_true=float(LT[:, gold].mean()),
                    gold_lin=float(LP[:, gold].mean()),
                    nt_mean=float(nt_.mean()),
                )
                if SMOKE:
                    npz_payload[key + '_cosS'] = cos_S[:32]
                log('  %s: cos_full=%.4f/%.4f cos_S=%.4f/%.4f band=%.4f '
                    'svd8=%.3f gold t/p=%.3f/%.3f' % (
                        key, mm[key]['cos_full_mean'], mm[key]['cos_full_med'],
                        mm[key]['cos_S_mean'], mm[key]['cos_S_med'],
                        mm[key]['band_med'], svd8,
                        mm[key]['gold_true'], mm[key]['gold_lin']))
                del LT, LP, d_true, dv, gprod, corr, jdv
    npz_payload[m + '_ustar'] = ustar
    npz_payload[m + '_med_abs'] = np.array(med_abs)
    out[m] = dict(design_sha8=DESIGN_SHA, device_anchors='drift69<1e-9, '
                  'drift89=0, drift90=0, drift91=0', metrics=mm,
                  wu_key=wu_key, wu_shape=list(W_U.shape), eps=EPS[m],
                  readout=kout, final_layer=NHz - 1)
    del z, H16, Y, H_final, Q_rows, Qn, W_U, gamma, tensors
    log('  %s done (%.1fs)' % (m, time.time() - T0))

# ---------- STEP 3: SEAL ----------
npz_path = os.path.join(OUTDIR, PFX + 'a1_behavioral_band.npz')
np.savez_compressed(npz_path, **npz_payload)
clean = True
verdict = 'a1_behavioral_band_sealed'
res = {
    'phase': 3225, 'name': NAME,
    'arm': 'A1 compute+seal (zero GPU compute; plan A weights, CPU matrix '
           'math only)',
    'prereg_authorization_record': '"好的，继续" = preregistered-default-arm '
                                   'trigger -> plan A (5.56GB minimal) '
                                   'downloaded; per-shard sha8 anchors '
                                   'verified at load time',
    'verdict': verdict,
    'created': datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
    'design_sha8': DESIGN_SHA,
    'models': {m: {'metrics': out[m]['metrics'],
                   'med_abs_vhat_held': [float(x) for x in npz_payload[m + '_med_abs']],
                   'device_anchors': out[m]['device_anchors'],
                   'wu_key': out[m]['wu_key'],
                   'wu_shape': out[m]['wu_shape']} for m in MODELS_A1},
    'sealed_npz': {'file': PFX + 'a1_behavioral_band.npz',
                   'file_sha8': sha8_file(npz_path)},
    'gate_evaluation': 'DEFERRED to 3226 (A2) per design chain',
    'contradictions': [],
}
with open(os.path.join(OUTDIR, PFX + 'result.json'), 'wb') as f:
    f.write(json.dumps(res, ensure_ascii=False, indent=1).encode('utf-8'))
with open(os.path.join(OUTDIR, PFX + 'run_log.txt'), 'w', encoding='utf-8') as f:
    f.write('\n'.join(LOG) + '\n')
log('== Phase 3225 %s %s ==' % (NAME, 'CLEAN' if clean else 'DIRTY'))
log('verdict: %s' % verdict)
