# -*- coding: utf-8 -*-
"""Phase 3230 (G5-G27): relation-contrast (state) direction efficacy arm.

Per prereg in MEMO Phase 3229 tail (frozen before any observation; default
arm (a), zero GPU): test whether the 3228/3229 anti-directional /
zero-efficacy finding is SPECIFIC to the T2 ctx-contrast recipe (u_R =
unit_mean of row-normalized ctx contrasts at slot KOUT=NL-1, 3188) by
building the row-level STATE contrast direction and re-running the exact
3228 flip protocol on it:
  dR = mean(h[isa rows]) - mean(h[hasa rows]) over the sealed 3157 panel
  (64 vs 64 rows, balanced), computed at the POST-NORM pre-logit slot
  NH-1 (= NL) -- the SAME slot/degree-of-freedom as the 3228 C9 injection
  site (space-coherent; the T2 u_R was computed at slot KOUT=NL-1, the
  raw last-layer output, per the 3188 recipe).
Protocol verbatim from sealed 3228 (freeze c51bfea5, res 99693298):
direct linear readout map l = W_U @ (h + dv); dv = unit(v) * alpha *
rms(h_row); alphas {0.5,1,2}; nulls = 10 pinned random seeds + orth
(dR projected out of row); panels 3154 seen (folds ci 0..3, ho=0) +
3169 OOV (templates t 0..2); judgment J = sign(m); flip toward R =
[J0 != R] & [J1 == R].
GATES (preregistered HERE, before any dR number is computed):
  GA_flip: flip_rate(+dR -> isa, alpha=2.0) - max(null flip rate at
      alpha=2.0, same slice) >= 0.15 on >= 3/4 seen folds x 2 models.
  GB_dir: seen dD(+dR) > 0 at alpha=1.0 AND strictly increasing across
      alpha 0.5 -> 1.0 -> 2.0, on both models.
  verdict dR_lever_pass iff GA_flip and GB_dir; else dR_lever_fail
      (routes to F-ledger refinement in the 3231 decision arm; not
      hidden).
DIAGNOSTICS (report-only, preregistered): cos(unit(dR), v_isa) /
cos(unit(dR), v_hasa) / cos(unit(dR), unit(W_isa - W_hasa)) /
cos(unit(dR), unit(dR_raw)). Pre-registered reading of cos(dR, v_isa):
>= +0.5 -> same-direction (anti-direction NOT recipe-explained, needs a
new account); <= -0.5 -> anti-parallel (recipe-specific confirmed, F20
refinement); |cos| < 0.5 -> the two recipes are decoupled, the dR result
stands alone. dD(-dR) symmetry and OOV dose/flip report-only.
DEVICE ANCHORS (frozen): 3157 tc_mean_cos <1e-9 (T2 slot kout); 3183
cos_rl/norm_dv <1e-9; C6v3 baseline marg argmax agreement >=0.98 per
model (3228 arbitrated 0.9963/0.9941); RMS scale sanity median
RMS(slot NH-1) < 5 < median RMS(slot NH-2) on 3157 (probed 2.16/11.98
and 2.19/25.31).
K1 clarification (freeze-time, vs the 3229 prereg wording): dR is read
at slot NH-1 (post-norm, C9 space coherence); the prereg phrase "3157
KOUT 缓存均值" is honored by ALSO sealing dR_raw at slot KOUT=NL-1.
Ledger read-only in this arm (n=381 post b21db7e3); registry changes
deferred to the 3231 decision arm.
SMOKE (P3230_SMOKE=1): qwen3-4b only, identical numbers (zero GPU).
"""
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
import phase3194_g5d2_ftr26_heldout_upgrade as p94  # sealed recipes

ROOT = p94.ROOT
SRC_DIR = p94.SRC_DIR
NAME = 'g5g27_relcontrast_efficacy'
OUTDIR = os.path.join(SRC_DIR, 'phase3230', NAME)
SMOKE = os.environ.get('P3230_SMOKE') == '1'
PFX = 'smoke_' if SMOKE else ''
MODELS = ['qwen3-4b'] if SMOKE else ['qwen3-4b', 'qwen3-14b']

LOG = []


def log(s):
    LOG.append(s)
    print(s, flush=True)


def sha8_file(p):
    return hashlib.sha256(open(p, 'rb').read()).hexdigest()[:8]


def csha8(obj):
    return hashlib.sha256(json.dumps(
        obj, ensure_ascii=False, indent=1, sort_keys=True).encode('utf-8')
    ).hexdigest()[:8]


os.makedirs(OUTDIR, exist_ok=True)
bad = []

ALPHAS = (0.5, 1.0, 2.0)
NULL_SEEDS = [20260390, 20260391, 20260392, 20260393, 20260394,
              20260395, 20260396, 20260397, 20260398, 20260399]
TOK_ISA = 101158   # '是一种' single token (sealed 3169 TPL0/TPL2 + 3157 isa)
TOK_HASA = 18830   # '有' (sealed 3157 hasa template)
CLS_NAMES = ['水果', '动物', '交通工具', '家具', '金属', '颜色',
             '乐器', '天气', '运动', '电器']
HUB = os.path.join(os.path.expanduser('~'), '.cache', 'huggingface', 'hub')
WU_SOURCES = {
    'qwen3-4b': dict(
        path=os.path.join(ROOT, 'models', 'hf', 'qwen3-4b',
                          'model-00001-of-00003.safetensors'),
        key='model.embed_tokens.weight', sha8='328a91d3',
        provenance='LOCAL collector weights (C9 provenance correction, 3228)'),
    'qwen3-14b': dict(
        path=glob.glob(os.path.join(HUB, 'models--Qwen--Qwen3-14B',
                                    'snapshots', '*'))[0]
             + os.sep + 'model-00008-of-00008.safetensors',
        key='lm_head.weight', sha8='0d6b9229',
        provenance='hub snapshot == local collector weights (C9, 3228)'),
}
CACHE_3154 = {'qwen3-4b': 'bb99438f', 'qwen3-14b': '6d142b9d'}
CACHE_3169 = {'qwen3-4b': '36eb4ff0', 'qwen3-14b': '97f98575'}
CACHE_3157 = {'qwen3-4b': '95a25965', 'qwen3-14b': '9552086d'}
LED_N, LED_POST = 381, 'b21db7e3'
GATE_FLIP_THRESH = 0.15
GATE_FLIP_FOLDS = 3

# ---------- STEP 0: device facts (token ids, before freeze) ----------
from tokenizers import Tokenizer
SNAP = {'qwen3-4b': glob.glob(os.path.join(HUB, 'models--Qwen--Qwen3-4B-Base', 'snapshots', '*'))[0],
        'qwen3-14b': glob.glob(os.path.join(HUB, 'models--Qwen--Qwen3-14B', 'snapshots', '*'))[0]}
TOKPATH = {'qwen3-4b': os.path.join(ROOT, 'models', 'hf', 'qwen3-4b', 'tokenizer.json'),
           'qwen3-14b': os.path.join(SNAP['qwen3-14b'], 'tokenizer.json')}
TOK = {}
for m in MODELS:
    tk = Tokenizer.from_file(TOKPATH[m])
    TOK[m] = tk
    assert tk.encode('是一种', add_special_tokens=False).ids == [TOK_ISA], (m, 'isa token')
    assert tk.encode('有', add_special_tokens=False).ids == [TOK_HASA], (m, 'hasa token')
    assert TOK_ISA != TOK_HASA
    cls_tok = [tk.encode(c, add_special_tokens=False).ids[0] for c in CLS_NAMES]
    assert len(set(cls_tok)) == 10, ('D2 distinctness replication', m, cls_tok)
    log('STEP0 %s tokens OK: isa=%d hasa=%d CLS_TOK distinct x10 (D2 replication)'
        % (m, TOK_ISA, TOK_HASA))
CLS_TOK = {m: [TOK[m].encode(c, add_special_tokens=False).ids[0] for c in CLS_NAMES]
           for m in MODELS}

# ---------- STEP 1: FREEZE (before any observation) ----------
fz = {
    'phase': '3230', 'name': NAME,
    'prereg': 'AGI_GPT5_MEMO Phase 3229 tail (frozen before any observation; '
              'default arm (a) relation-contrast efficacy, zero GPU compute)',
    'models': MODELS,
    'question': 'is the 3228/3229 anti-directional / zero-efficacy finding '
                'SPECIFIC to the T2 ctx-contrast recipe (u_R), or does it '
                'generalize to the row-level state-contrast direction dR?',
    'direction_dR': {
        'formula': 'dR = mean(h[isa rows]) - mean(h[hasa rows]) over the '
                   'sealed 3157 panel (rel tag 0=isa / 1=hasa; 64 vs 64 rows, '
                   'balanced; includes all pol x ctx cells), H fp32',
        'slot': 'NH-1 (= NL): the POST-NORM pre-logit state — SAME slot as '
                'the 3228 C9 injection site (space-coherent)',
        'K1_clarification': 'the 3229 prereg wording "3157 KOUT 缓存均值" is '
                            'read at slot NH-1 per C9 space coherence; the '
                            'KOUT=NL-1 raw-slot version is ALSO sealed as '
                            'descriptive dR_raw (cos(dR, dR_raw) reported)',
        'normalize': 'unit(dR) for injection; dv = unit(dR) * alpha * '
                     'rms(h_row) [3228 C4/C9 convention]',
        'sign_convention': '+dR points from hasa-mean toward isa-mean; +dR '
                           'injection should INCREASE the isa margin '
                           '(m = logit_isa - logit_hasa) if the state '
                           'contrast is the behavioral lever',
    },
    'panels': {
        'seen_3154': {'file': r'phase3154\g1p4_mfd_multifactor_disentangle\{m}\collect.npz',
                      'sha8': CACHE_3154, 'rows': 208,
                      'final_slot': 'index NH-1 (= NL): 36 (4b) / 40 (14b) — '
                                    'post-norm pre-logit state [C9]',
                      'folds': 'ci in {0,1,2,3} over train rows (ho=0), 48 '
                               'rows each; heldout ho=1 not used in this arm',
                      'tags': ['g', 'l', 's', 'ti', 'ci', 'ho']},
        'oov_3169': {'file': r'phase3169\g5a6_oov_panel\collect_{m}.npz',
                     'sha8': CACHE_3169, 'rows': 2190,
                     'final_slot': 'index NH-1: 36 (4b) / 40 (14b)',
                     'slices': 'template t in {0,1,2}',
                     'tags': ['t', 'pi', 'pair(i,c)', 'is_oov_class', 'oov_class']},
        'directions_3157': {'file': r'phase3157\g2p2_transform_algebra_commutator\{m}\collect.npz',
                            'sha8': CACHE_3157, 'rows': 128,
                            'tags': 'ei/rel(0=isa,1=hasa)/pol/ctx',
                            'slots_used': 'NH-1 (dR, primary) + KOUT=NL-1 '
                                          '(dR_raw descriptive + T2 u_R '
                                          'recipe anchors)'},
    },
    'protocol_inherited': 'flip protocol verbatim from sealed 3228 (freeze '
                          'c51bfea5, res 99693298): direct linear map '
                          'l = W_U @ (h + dv); J = sign(m); flip toward R = '
                          '[J0 != R] & [J1 == R]; slices seen cut0-3 (48 '
                          'rows, ci & ho=0) + oov tpl0-2 (730 each); alphas '
                          '{0.5,1,2}; nulls 10 pinned seeds + orth',
    'perts': ['dR (+, real)', 'dR_neg (-, control)',
              'rand_20260390..99 (null)', 'orth_dR (null)'],
    'baseline_context_sealed_3228': {
        'seen_d0_mean': {'qwen3-4b': -3.966, 'qwen3-14b': -4.693},
        'seen_J0_isa_share': 0.0,
        'oov_d0_mean': {'qwen3-4b': -2.241, 'qwen3-14b': -1.744},
        'note': 'sealed PRIOR numbers used for gate design rationale only; '
                'no new observation before this freeze'},
    'gates': {
        'GA_flip': 'flip_rate(+dR -> isa, alpha=2.0) - max(null flip rate at '
                   'alpha=2.0, same slice) >= 0.15 on >= 3/4 seen folds x 2 '
                   'models',
        'GB_dir': 'seen dD(+dR) > 0 at alpha=1.0 AND strictly increasing '
                  'across alpha 0.5 -> 1.0 -> 2.0, on both models',
        'verdict': 'dR_lever_pass iff GA_flip and GB_dir; else dR_lever_fail '
                   '(routes to F-ledger refinement in the 3231 decision '
                   'arm; not hidden)',
    },
    'diagnostics_report_only': {
        'align': 'cos(unit(dR), v_isa) / cos(unit(dR), v_hasa) / '
                 'cos(unit(dR), unit(W_isa - W_hasa)) / cos(unit(dR), '
                 'unit(dR_raw))',
        'align_reading_prereg': 'cos(dR, v_isa) >= +0.5 -> same-direction '
                                '(anti-direction NOT recipe-explained, needs '
                                'a new account); <= -0.5 -> anti-parallel '
                                '(recipe-specific confirmed, F20 refinement); '
                                '|cos| < 0.5 -> recipes decoupled, the dR '
                                'result stands alone',
        'dose_symmetry': 'dD(-dR) reported (sign mirror expected under the '
                         'linear map)',
        'oov': 'OOV dose + flips report-only (gates bind seen folds)',
    },
    'device_gates': {
        'd_3157_tc': 'tc_mean_cos anchor <1e-9 per model (T2 slot kout)',
        'd_3183_dv': 'cos_rl/norm_dv replication <1e-9 per model',
        'd_c6v3_marg': 'baseline direct-map class readout vs sealed 3169 '
                       'marg: argmax agreement >= 0.98 per model (3228 '
                       'arbitrated 0.9963 local-4B / 0.9941 14B)',
        'd_rms_scale': '3157 median RMS(slot NH-1) < 5 < median RMS(slot '
                       'NH-2) per model (C9 post-norm scale; probed 2.16/'
                       '11.98 and 2.19/25.31)',
    },
    'weights': {
        'map': 'l = W_U @ (h + dv), h = cached pre-logit state; exactly '
               'linear [3228 C9]',
        'sources': {m: dict(path=WU_SOURCES[m]['path'].replace(ROOT, ''),
                            key=WU_SOURCES[m]['key'], sha8=WU_SOURCES[m]['sha8'],
                            provenance=WU_SOURCES[m]['provenance'])
                    for m in MODELS},
    },
    'ledger': 'read-only in this arm (n=381 post b21db7e3); registry changes '
              'deferred to the 3231 decision arm',
    'zero_gpu': 'CPU matrix math only; no model forward, no GPU',
}
fz['sha8'] = csha8(dict(fz))
with open(os.path.join(OUTDIR, PFX + 'execution.json'), 'w',
          encoding='utf-8', newline='') as f:
    json.dump(fz, f, ensure_ascii=False, indent=1)
log('== Phase 3230 %s %s == freeze %s (before any dR observation)'
    % (NAME, 'SMOKE' if SMOKE else 'FULL', fz['sha8']))

# ---------- sealed inputs + ledger snapshot ----------
led0 = json.load(open(os.path.join(ROOT, 'research', 'gpt5', 'atlas',
                                   'atlas_ledger.json'), encoding='utf-8'))
assert len(led0['measurements']) == LED_N and led0['ledger_sha256_8'] == LED_POST
log('ledger snapshot: n=%d post=%s (read-only arm)' % (LED_N, LED_POST))
sealed_3183 = json.load(open(p94.SRC['p3183_result'], encoding='utf-8'))
assert sealed_3183['res_sha8'] == 'c55e4e5e' and sealed_3183['seal_sha8'] == '7196fa8a'
sealed_3169 = json.load(open(p94.SRC['p3169_result'], encoding='utf-8'))
CFG = {m: json.load(open(os.path.join(SNAP[m], 'config.json'), encoding='utf-8'))
       for m in MODELS}
EPS = {m: float(CFG[m].get('rms_norm_eps', 1e-6)) for m in MODELS}

PANEL_3154 = {
    'qwen3-4b': os.path.join(SRC_DIR, r'phase3154\g1p4_mfd_multifactor_disentangle\qwen3-4b\collect.npz'),
    'qwen3-14b': os.path.join(SRC_DIR, r'phase3154\g1p4_mfd_multifactor_disentangle\qwen3-14b\collect.npz'),
}
PANEL_3169 = {m: p94.SRC[p94.NPZ69_KEY[m]] for m in MODELS}

out = {}
npz_payload = {}
gate_summary = {}
for m in MODELS:
    log('=== %s ===' % m)
    D = int(CFG[m]['hidden_size'])
    NL = int(CFG[m]['num_hidden_layers'])
    NH = NL + 1
    fin = NH - 1

    # --- 3157 cache: dR (post-norm slot) + dR_raw (KOUT slot) + T2 anchors ---
    p57 = p94.SRC[p94.NPZ57_KEY[m]]
    assert sha8_file(p57) == CACHE_3157[m], ('3157 sha', m)
    z57 = np.load(p57)
    H57 = z57['H'].astype(np.float32)
    assert H57.shape == (128, NH, D), ('3157 shape', m, H57.shape)
    EI = z57['ei'].astype(int)
    RL = z57['rel'].astype(int)
    PL = z57['pol'].astype(int)
    CX = z57['ctx'].astype(int)
    del z57
    assert set(np.unique(RL).tolist()) == {0, 1}
    isa_mask = (RL == 0)
    hasa_mask = (RL == 1)
    assert isa_mask.sum() == 64 and hasa_mask.sum() == 64, ('3157 balance', m)
    # RMS scale sanity (C9 post-norm slot)
    med_fin = float(np.median(np.sqrt((H57[:, fin, :] ** 2).mean(1))))
    med_prev = float(np.median(np.sqrt((H57[:, fin - 1, :] ** 2).mean(1))))
    assert med_fin < 5.0 < med_prev, ('3157 RMS scale', m, med_fin, med_prev)
    log('  3157 RMS scale OK: med[NH-1]=%.3f med[NH-2]=%.3f' % (med_fin, med_prev))
    # dR primary (post-norm pre-logit slot, C9-coherent)
    dR_post = H57[:, fin, :][isa_mask].mean(0) - H57[:, fin, :][hasa_mask].mean(0)
    dR_raw = H57[:, fin - 1, :][isa_mask].mean(0) - H57[:, fin - 1, :][hasa_mask].mean(0)
    norm_post = float(np.linalg.norm(dR_post))
    norm_raw = float(np.linalg.norm(dR_raw))
    dR_unit = (dR_post / (norm_post + 1e-18)).astype(np.float32)
    dR_raw_unit = (dR_raw / (norm_raw + 1e-18)).astype(np.float32)
    assert np.isfinite(dR_unit).all() and np.isfinite(dR_raw_unit).all()
    cos_dr_raw = float(dR_unit @ dR_raw_unit)
    log('  dR: |dR_post|=%.4f |dR_raw|=%.4f cos(dR,dR_raw)=%.4f'
        % (norm_post, norm_raw, cos_dr_raw))

    # T2 u_R recipe verbatim (3228 lines) + anchors
    A57 = p94.ANCH_3157[p94.ANCH_KEY[m]]
    IDX = {(int(EI[i]), int(RL[i]), int(PL[i]), int(CX[i])): i for i in range(128)}
    assert len(IDX) == 128
    dC = []
    for ei in range(16):
        for rl in range(2):
            for pl in range(2):
                dC.append(H57[IDX[(ei, rl, pl, 1)], A57['kout']]
                          - H57[IDX[(ei, rl, pl, 0)], A57['kout']])
    dC = np.array(dC)
    assert np.isfinite(dC).all()
    dCn = dC / (np.linalg.norm(dC, axis=1, keepdims=True) + 1e-18)
    cosM = dCn @ dCn.T
    tri = cosM[np.triu_indices(len(dC), 1)]
    tc_full = float(tri.mean())
    d_tc = abs(tc_full - A57['tc_mean_cos'])
    assert d_tc < 1e-9, ('3157 tc anchor', m, d_tc)
    RL_rows = np.array([(i // 4, (i % 4) // 2, i % 2) for i in range(64)])
    v_isa = p94.unit_mean(dCn[RL_rows[:, 1] == 0]).astype(np.float32)
    v_hasa = p94.unit_mean(dCn[RL_rows[:, 1] == 1]).astype(np.float32)
    cos_rl = float(v_isa @ v_hasa)
    norm_dv = float(np.linalg.norm(v_isa - v_hasa))
    E83 = sealed_3183['per_model'][m]
    d_83 = max(abs(cos_rl - E83['cos_rl']), abs(norm_dv - E83['norm_dv']))
    assert d_83 < 1e-9, ('3183 dv anchor', m, d_83)
    log('  u_R anchors OK: tc drift %.2e, 3183 drift %.2e (cos_rl=%.6f norm_dv=%.6f)'
        % (d_tc, d_83, cos_rl, norm_dv))
    del H57, dC, dCn, cosM

    # --- weights (C9 provenance sources) ---
    from safetensors import safe_open
    src = WU_SOURCES[m]
    assert sha8_file(src['path']) == src['sha8'], ('W_U sha', m, src['path'])
    with safe_open(src['path'], framework='pt') as f:
        avail = list(f.keys())
        assert src['key'] in avail, ('key missing', m, src['key'])
        W_U = f.get_tensor(src['key']).float().numpy()
    assert W_U.shape == (int(CFG[m]['vocab_size']), D), ('W_U shape', m, W_U.shape)
    WT = W_U[[TOK_ISA, TOK_HASA]].astype(np.float32)
    WUC = W_U[CLS_TOK[m]].astype(np.float32)
    del W_U
    wdw = WT[0] - WT[1]
    wdw = wdw / (float(np.linalg.norm(wdw)) + 1e-18)

    # --- diagnostics (report-only, computed on sealed caches) ---
    diag = dict(cos_dR_visa=float(dR_unit @ v_isa),
                cos_dR_vhasa=float(dR_unit @ v_hasa),
                cos_dR_wdw=float(dR_unit @ wdw),
                cos_dR_dRraw=cos_dr_raw,
                norm_post=norm_post, norm_raw=norm_raw)
    log('  diag: cos(dR,v_isa)=%+.4f cos(dR,v_hasa)=%+.4f '
        'cos(dR,W_isa-W_hasa)=%+.4f'
        % (diag['cos_dR_visa'], diag['cos_dR_vhasa'], diag['cos_dR_wdw']))

    # --- flip protocol (3228 verbatim, perts = dR family) ---
    def flip_block(tag, Hrows, slices, slice_names, npz_prefix, perts):
        n = Hrows.shape[0]
        rms = np.sqrt((Hrows ** 2).mean(axis=1, keepdims=True))
        m0 = Hrows @ WT.T                     # (n, 2) primary margins
        c0 = Hrows @ WUC.T                    # (n, 10) class readout
        J0 = (m0[:, 0] > m0[:, 1]).astype(np.int8)
        n_tie = int((m0[:, 0] == m0[:, 1]).sum())
        if n_tie:
            bad.append('tie baseline %s n=%d' % (tag, n_tie))
        marg_diag = None
        if tag == 'oov':
            cached = MARG69.reshape(3 * 730, 10)
            am_mine = c0.argmax(1)
            am_cached = cached.argmax(1)
            agree = (am_mine == am_cached)
            g = np.sort(cached, axis=1)
            gap = g[:, 9] - g[:, 8]
            gm = float(np.mean([float(np.corrcoef(
                (c0[i] - c0[i].mean()), cached[i])[0, 1]) for i in range(0, n, 16)]))
            marg_diag = dict(agree_all=float(agree.mean()),
                             agree_gap05=float(agree[gap >= 0.5].mean()),
                             agree_gap10=float(agree[gap >= 1.0].mean()),
                             mean_centered_cos=gm,
                             n_gap05=int((gap >= 0.5).sum()),
                             n_gap10=int((gap >= 1.0).sum()))
            log('  %s C6v3 gate: agree all=%.4f (floor 0.98) gap>=0.5=%.4f '
                '(n=%d) centered-cos=%.4f'
                % (tag, marg_diag['agree_all'], marg_diag['agree_gap05'],
                   marg_diag['n_gap05'], gm))
            if marg_diag['agree_all'] < 0.98:
                bad.append('C6v3 floor %s=%.4f' % (m, marg_diag['agree_all']))
        margins = np.zeros((len(perts), len(ALPHAS), n, 2), np.float32)
        for pi_, (pname, vunit, kind) in enumerate(perts):
            for ai_, a in enumerate(ALPHAS):
                dv = vunit * (a * rms)
                Hp = Hrows + dv
                margins[pi_, ai_] = Hp @ WT.T
                del Hp, dv
        agg = {}
        for si_, sl in enumerate(slices):
            for R, Rname in ((1, 'isa'), (0, 'hasa')):
                for pi_, (pname, _, kind) in enumerate(perts):
                    for ai_, a in enumerate(ALPHAS):
                        J1 = (margins[pi_, ai_, :, 0] > margins[pi_, ai_, :, 1])
                        fl = (J0[sl] == (1 - R)) & (J1[sl].astype(np.int8) == R)
                        nr = int(len(sl))
                        nf = int(fl.sum())
                        agg['%s|%s|%s|a%s' % (Rname, pname, slice_names[si_], a)] = \
                            dict(n=nf, rows=nr, rate=float(nf / nr))
        npz_payload[npz_prefix + '_margins'] = margins
        npz_payload[npz_prefix + '_J0'] = J0
        npz_payload[npz_prefix + '_m0'] = m0
        dose = {}
        for pi_, (pname, _, kind) in enumerate(perts):
            for ai_, a in enumerate(ALPHAS):
                d1 = margins[pi_, ai_, :, 0] - margins[pi_, ai_, :, 1]
                dose['%s|a%s' % (pname, a)] = float((d1 - (m0[:, 0] - m0[:, 1])).mean())
        return dict(m0_isa=float(m0[:, 0].mean()), m0_hasa=float(m0[:, 1].mean()),
                    J0_isa_share=float(J0.mean()), n_rows=n, n_tie=n_tie,
                    d0_mean=float((m0[:, 0] - m0[:, 1]).mean()),
                    d0_std=float((m0[:, 0] - m0[:, 1]).std()),
                    marg_diag=marg_diag, dose=dose, agg=agg,
                    perts=[p[0] for p in perts])

    # perts: dR family + 10 pinned randoms + orth(dR)
    perts = [('dR', np.tile(dR_unit, (1, 1)), 'real'),
             ('dR_neg', np.tile(-dR_unit, (1, 1)), 'control')]
    for s in NULL_SEEDS:
        rng = np.random.RandomState(s)
        v = rng.randn(D).astype(np.float32)
        v /= float(np.linalg.norm(v))
        perts.append(('rand_%d' % s, v[None, :], 'null'))
    # orth_dR built per row inside materialize_perts (row-dependent).
    perts.append(('orth_dR', None, 'orth'))

    def materialize_perts(Hrows):
        mats = []
        for pname, vunit, kind in perts:
            if kind == 'orth':
                unh = dR_unit
                pr = (Hrows @ unh)[:, None] * unh[None, :]
                wo = Hrows - pr
                wo = wo / (np.linalg.norm(wo, axis=1, keepdims=True) + 1e-18)
                mats.append((pname, wo, kind))
            else:
                mats.append((pname, np.tile(vunit, (Hrows.shape[0], 1))
                             if vunit.shape[0] == 1 else vunit, kind))
        return mats

    # seen panel
    z54 = np.load(PANEL_3154[m])
    H54 = z54['H']
    assert H54.shape == (208, NH, D), ('3154 shape', m, H54.shape)
    tags54 = {k: z54[k].astype(int) for k in ('ci', 'ho')}
    del z54
    H54f = H54[:, fin, :].astype(np.float32)
    del H54
    sls54 = [np.where((tags54['ci'] == c) & (tags54['ho'] == 0))[0] for c in range(4)]
    assert all(len(s) == 48 for s in sls54), ('fold sizes', m, [len(s) for s in sls54])
    perts_seen = materialize_perts(H54f)
    seen_res = flip_block('seen', H54f, sls54, ['cut%d' % c for c in range(4)],
                          m + '_seen', perts_seen)
    del H54f

    # OOV panel
    rd = int(sealed_3169['per_model'][m]['readout'])
    assert rd == NL - 1, ('readout slot', m)
    z69 = np.load(PANEL_3169[m])
    H69 = z69['H']
    MARG69 = z69['marg']
    assert H69.shape == (3, 730, NH, D), ('3169 shape', m, H69.shape)
    H69_final = H69[:, :, fin, :].reshape(3 * 730, D).astype(np.float32)
    del z69, H69
    sls69 = [np.arange(t * 730, (t + 1) * 730) for t in range(3)]
    perts_oov = materialize_perts(H69_final)
    oov_res = flip_block('oov', H69_final, sls69, ['tpl%d' % t for t in range(3)],
                         m + '_oov', perts_oov)
    del H69_final, MARG69

    # summary log: matched flip per fold at each alpha (seen)
    for a in ('0.5', '1.0', '2.0'):
        for f in ('cut0', 'cut1', 'cut2', 'cut3'):
            matched = seen_res['agg']['isa|dR|%s|a%s' % (f, a)]['rate']
            nullmax = max(vv['rate'] for kk, vv in seen_res['agg'].items()
                          if kk.endswith('|%s|a%s' % (f, a))
                          and kk.split('|')[1].startswith(('rand_', 'orth_')))
            log('  seen a%s %s: dR_flip=%.4f nullmax=%.4f diff=%+.4f'
                % (a, f, matched, nullmax, matched - nullmax))
    log('  dose (seen): %s' % '; '.join(
        '%s a%s dD=%+.4f' % (pn, a, seen_res['dose']['%s|a%s' % (pn, a)])
        for pn in ('dR', 'dR_neg') for a in ('0.5', '1.0', '2.0')))
    log('  dose (oov): %s' % '; '.join(
        '%s a%s dD=%+.4f' % (pn, a, oov_res['dose']['%s|a%s' % (pn, a)])
        for pn in ('dR', 'dR_neg') for a in ('0.5', '1.0', '2.0')))

    # --- gates (frozen definitions) ---
    ga_folds = 0
    ga_detail = {}
    for f in ('cut0', 'cut1', 'cut2', 'cut3'):
        matched = seen_res['agg']['isa|dR|%s|a2.0' % f]['rate']
        nullmax = max(vv['rate'] for kk, vv in seen_res['agg'].items()
                      if kk.endswith('|%s|a2.0' % f)
                      and kk.split('|')[1].startswith(('rand_', 'orth_')))
        diff = matched - nullmax
        ga_detail[f] = dict(matched=matched, nullmax=nullmax, diff=diff)
        if diff >= GATE_FLIP_THRESH:
            ga_folds += 1
    ga_pass = ga_folds >= GATE_FLIP_FOLDS
    dd = seen_res['dose']
    gb_vals = (dd['dR|a0.5'], dd['dR|a1.0'], dd['dR|a2.0'])
    gb_pass = bool(dd['dR|a1.0'] > 0 and gb_vals[0] < gb_vals[1] < gb_vals[2])
    gate_summary[m] = dict(ga_folds=ga_folds, ga_pass=ga_pass,
                           ga_detail=ga_detail, gb_pass=gb_pass,
                           gb_vals=[float(x) for x in gb_vals])
    log('  GATES %s: GA_flip folds=%d/4 (>=3 needed) -> %s | GB_dir '
        'dD=%s -> %s' % (m, ga_folds, ga_pass,
                         '/'.join('%+.4f' % x for x in gb_vals), gb_pass))

    out[m] = dict(
        dR=dict(norm_post=norm_post, norm_raw=norm_raw, diag=diag),
        u_R_anchors=dict(tc_drift=d_tc, dv_drift=d_83, cos_rl=cos_rl,
                         norm_dv=norm_dv),
        device=dict(rms_scale=dict(med_fin=med_fin, med_prev=med_prev),
                    marg_diag=oov_res['marg_diag']),
        seen=seen_res, oov=oov_res, gates=gate_summary[m],
    )
    npz_payload[m + '_dR_unit'] = dR_unit
    npz_payload[m + '_dR_raw_unit'] = dR_raw_unit
    log('  %s done (%.1fs)' % (m, time.time() - T0))

# ---------- STEP 3: global gates + SEAL ----------
GA = all(gate_summary[m]['ga_pass'] for m in MODELS)
GB = all(gate_summary[m]['gb_pass'] for m in MODELS)
clean = not bad
if GA and GB and clean:
    verdict = 'dR_lever_pass'
elif GA and GB:
    verdict = 'dR_lever_pass_dirty(%d)' % len(bad)
else:
    verdict = 'dR_lever_fail'
    if not clean:
        verdict = verdict + '_dirty(%d)' % len(bad)

npz_path = os.path.join(OUTDIR, PFX + 'e3_relcontrast_flip.npz')
np.savez_compressed(npz_path, **npz_payload)
led1 = json.load(open(os.path.join(ROOT, 'research', 'gpt5', 'atlas',
                                   'atlas_ledger.json'), encoding='utf-8'))
assert len(led1['measurements']) == LED_N and led1['ledger_sha256_8'] == LED_POST
log('ledger unchanged: n=%d post=%s before==after (read-only arm)'
    % (LED_N, LED_POST))

res = {
    'phase': 3230, 'name': NAME,
    'arm': 'relation-contrast (state) direction efficacy (default (a), '
           'zero GPU compute, CPU matrix math only)',
    'prereg_authorization_record': '"好的，继续" = preregistered-default-arm trigger',
    'freeze_sha8': fz['sha8'],
    'question': fz['question'],
    'clarifications': [
        'K1 dR computed at slot NH-1 (post-norm pre-logit, C9 space '
        'coherence with the injection site); prereg wording "3157 KOUT 缓存'
        '均值" honored by also sealing dR_raw at slot KOUT=NL-1 (cos '
        'reported)',
        'K2 dR sign: +dR points hasa-mean -> isa-mean; +dR should increase '
        'the isa margin if the state contrast is the lever',
        'K3 protocol/panels/nulls/alphas inherited verbatim from sealed '
        '3228 (freeze c51bfea5, res 99693298)',
        'K4 gates GA_flip/GB_dir frozen BEFORE any dR number was computed; '
        'baseline context taken from sealed 3228 numbers only',
        'K5 registry changes deferred to the 3231 decision arm; ledger '
        'read-only here'],
    'models': {m: dict(
        dR=out[m]['dR'],
        u_R_anchors=out[m]['u_R_anchors'],
        device=out[m]['device'],
        gates=out[m]['gates'],
        seen=dict(n_rows=out[m]['seen']['n_rows'],
                  J0_isa_share=out[m]['seen']['J0_isa_share'],
                  d0_mean=out[m]['seen']['d0_mean'],
                  d0_std=out[m]['seen']['d0_std'],
                  marg_diag=out[m]['seen']['marg_diag'],
                  dose=out[m]['seen']['dose'],
                  agg=out[m]['seen']['agg']),
        oov=dict(n_rows=out[m]['oov']['n_rows'],
                 J0_isa_share=out[m]['oov']['J0_isa_share'],
                 d0_mean=out[m]['oov']['d0_mean'],
                 d0_std=out[m]['oov']['d0_std'],
                 marg_diag=out[m]['oov']['marg_diag'],
                 dose=out[m]['oov']['dose'],
                 agg=out[m]['oov']['agg']),
    ) for m in MODELS},
    'gates_global': {'GA_flip': GA, 'GB_dir': GB,
                     'per_model': gate_summary},
    'sealed_npz': {'file': PFX + 'e3_relcontrast_flip.npz',
                   'file_sha8': sha8_file(npz_path)},
    'contradictions': bad,
    'verdict': verdict,
    'res_sha8': None,
}
res['res_sha8'] = csha8(dict(res))
with open(os.path.join(OUTDIR, PFX + 'result.json'), 'w',
          encoding='utf-8', newline='') as f:
    f.write(json.dumps(res, ensure_ascii=False, indent=1))
with open(os.path.join(OUTDIR, PFX + 'run_log.txt'), 'w',
          encoding='utf-8', newline='') as f:
    f.write('\n'.join(LOG) + '\n')
log('RECHECK: %s (%d contradictions)' % ('CLEAN' if clean else 'DIRTY', len(bad)))
log('verdict: %s | res %s | npz %s'
    % (verdict, res['res_sha8'], res['sealed_npz']['file_sha8']))
