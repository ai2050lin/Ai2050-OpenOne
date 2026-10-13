# -*- coding: utf-8 -*-
"""Phase 3228 (G5-G26): criterion-2 E3 intervention — compute + seal arm.

Per prereg in MEMO Phase 3227 tail (frozen before any observation;
default arm (a), zero GPU compute): per sealed design a883c4da
(e3_intervention_design_v1.json), compute and seal
  (1) exact post-patch judgment matrices: l' = W_U @ ln_f(h + a*u) on
      CPU for every cached row x perturbation x alpha in {0.5,1,2};
      judgment J = sign(mean-logit isa-token-set - mean-logit
      hasa-token-set); flip toward R = [J_before = not R] and
      [J_after = R]  (design math verbatim);
  (2) null controls: random norm-matched x 10 pinned seeds +
      orthogonalized (u projected out of row);
  (3) GD_amp data: per-class spectral-out collapse (3169 B-recipe on
      cached panel) vs per-class u* amplitude (row projection onto
      sealed u*), rank correlation sealed for the 3229 gate.

FREEZE-time protocol pinning (clarifications vs the 3227 design doc,
each logged BEFORE any observation; design gate texts verbatim):
  C1. 3169 OOV-panel first dim is 3 TEMPLATES (design doc label "3
      seeds" corrected; on-disk H (3,730,NH,D) verified). OOV flip
      aggregation slices = template index t in {0,1,2}.
  C2. Token sets pinned from sealed prompt ids: isa-set = {101158}
      ("是一种", single token, appears verbatim in sealed 3169 TPL0/TPL2
      and 3157 isa template); hasa-set = {18830} ("有", 3157 hasa
      template "有{P}"). The design parenthetical "(3169 D2 first-token
      distinctness)" is honored: ids from sealed prompts, distinctness
      asserted (101158 != 18830), and the 3169 D2 CLS_TOK distinctness
      gate re-verified per model. Char-level pair margins (是 20412 vs
      有 18830) additionally sealed as descriptive; gates bind to the
      primary sets.
  C3. Seen-panel (3154) "4 folds" = the schema's only 4-way factor,
      the comma-cut groups ci in {0,1,2,3} over train rows (ho=0, 48
      rows each); held-out rows (ho=1, 16) sealed as descriptive extra.
  C4. Injection scale = A1 convention dv = unit(u_R) * alpha *
      rms(h_row) (design math "h + a*u" read at A1 row scale:
      the alphas {0.5,1,2} extend the A1 gate scale 0.5 and the design
      scope_limits bind alpha=2 to the A1 linear license, implying the
      A1 alpha convention). Nulls norm-matched to dv per row.
  C5. Orthogonalized null reading: v = unit(h_row - (h_row . u_R) u_R)
      (grammar of "u projected out of row"), norm-matched to dv.
  C6. Device anchor (C6v3, after two SMOKE-stage recalibrations, before
      the final FULL): the cached slot NH-1 is the POST-NORM pre-logit
      state (HF convention: the final normed state is appended after
      the layer loop; the raw last-layer output is NOT cached). The
      readout map from this state is EXACTLY LINEAR: l = W_U @ h.
      Evidence: baseline direct-map class readout vs the sealed 3169
      bf16 marg cache — argmax agreement 0.9963 (4B, local weights) /
      0.9941 (14B), centered-cos 1.0000 both. Gate: agreement >= 0.98
      per model. History: v1 pin (0.90 blanket) and v2 (0.80 floor +
      stratified report) were computed under the superseded
      double-norm map lnf(h_fin); the bf16-emulation arbitration
      (fp32-vs-bf16emul 0.9927, both-vs-cached 0.55) exposed the map
      error, and the C9 probe (direct vs double vs lnf(h[NH-2]))
      settled it.
  C9. Map + provenance correction (SMOKE-stage, pre-FULL): (i) readout
      map corrected per C6v3 — the design math "l' = W_U @ ln_f(h +
      a*u)" is read with h = the cached pre-logit state, whose ln_f
      factor is ALREADY absorbed in the cache; the pre-logit -> logit
      map is exactly linear (coherent with GAP-5: the nonlinearity
      lives upstream of this state); (ii) W_U provenance split found:
      the 4B caches were collected with the LOCAL model
      models/hf/qwen3-4b (shard1 328a91d3) whose weights DIFFER from
      the hub Qwen3-4B-Base snapshot (4c807e25) used by 3225 plan A;
      the corrected arm loads the LOCAL collector weights for 4B and
      the hub shard8 for 14B (0d6b9229 == local, exact-match proven).
      Cross-phase note (recorded for 3229): 3225 A1's behavioral-band
      numbers used the hub 4B W_U on local-collected residuals; A1's
      linearity conclusion (a property of the ln_f map) is unaffected,
      but the provenance caveat travels with the registry v1.29
      E2_behavioral upgrades.
  C8. SMOKE finding recorded (pre-FULL, under the superseded
      double-norm map; the corrected map re-measures): matched flip
      rates were 0 on all seen folds; u_isa shifted the isa/hasa
      margin anti-directionally and monotonically in alpha while
      u_hasa was ~50x weaker; the judgment itself had power (baseline
      margin std 3.8, both sides populated 55/45). The arm seals the
      corrected matrices as-is; gate application stays with 3229
      (honest FAIL routing per the design taxonomy).
  C7. GD_amp statistic pinned: per-class collapse = mean over class
      rows x seeds 7/8/9 of 3169-B normalized MSE; per-class amplitude
      = median |Qn_row . u*|; gate statistic = Spearman(amplitude,
      collapse) >= 0.6 (positive sign pinned).
E3 taxonomy honesty (design verbatim): in-protocol causality only —
last-layer residual injection bypasses upstream computation; no
full-regeneration claim. Gate APPLICATION (GD_*) is deferred to 3229;
this arm seals matrices + null controls + amplitude correlations.
SMOKE (P3228_SMOKE=1): qwen3-4b only, identical numbers (zero GPU).
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
import phase3194_g5d2_ftr26_heldout_upgrade as p94  # sealed recipes

ROOT = p94.ROOT
SRC_DIR = p94.SRC_DIR
NAME = 'g5g26_e3_compute_seal'
OUTDIR = os.path.join(SRC_DIR, 'phase3228', NAME)
SMOKE = os.environ.get('P3228_SMOKE') == '1'
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
TOK_SHI = 20412    # '是'  (descriptive char pair)
TOK_YOU = 18830    # '有'  (descriptive char pair, same as hasa set)
CLS_NAMES = ['水果', '动物', '交通工具', '家具', '金属', '颜色',
             '乐器', '天气', '运动', '电器']
WEIGHT_FILES = None  # superseded by WU_SOURCES (C9 provenance correction)
HUB = os.path.join(os.path.expanduser('~'), '.cache', 'huggingface', 'hub')
WU_SOURCES = {
    'qwen3-4b': dict(
        path=os.path.join(ROOT, 'models', 'hf', 'qwen3-4b',
                          'model-00001-of-00003.safetensors'),
        key='model.embed_tokens.weight', sha8='328a91d3',
        provenance='LOCAL collector weights (models/hf/qwen3-4b, tied '
                   'embed=lm_head); the hub Qwen3-4B-Base snapshot '
                   '(4c807e25) is a DIFFERENT revision — split found by '
                   'the C9 arbitration'),
    'qwen3-14b': dict(
        path=glob.glob(os.path.join(HUB, 'models--Qwen--Qwen3-14B',
                                    'snapshots', '*'))[0]
             + os.sep + 'model-00008-of-00008.safetensors',
        key='lm_head.weight', sha8='0d6b9229',
        provenance='hub snapshot == local collector weights (sha8 '
                   '0d6b9229 identical; C9 probe direct-map marg '
                   'agreement 0.9941, centered-cos 1.0000)'),
}
SHARD_SHA8 = None
CACHE_3154 = {'qwen3-4b': 'bb99438f', 'qwen3-14b': '6d142b9d'}
CACHE_3169 = {'qwen3-4b': '36eb4ff0', 'qwen3-14b': '97f98575'}
GATES_3227_VERBATIM = {
    'GD_flip': 'flip_rate(+u_R, alpha=1) - max(flip_rate over nulls) >= '
               '0.15 on >= 3/4 folds x 2 models (per relation direction)',
    'GD_dose': 'flip_rate monotone non-decreasing in alpha {0.5, 1, 2} '
               '(all pairwise increases or Spearman == 1.0) x 2 models',
    'GD_spec': 'relation specificity: cross-patch flip rate (patch u_isa, '
               'measure hasa judgment, and vice versa) <= 1/3 of matched '
               'rate x 2 models',
    'GD_amp': 'u* amplitude (row projection onto sealed u*) rank-correlates '
              'with measured per-class spectral-out collapse (3169 recipe '
              'on cached panel) >= 0.6 across OOV classes x 2 models '
              '(n=4 classes - small-n scope limit)',
    'verdict_taxonomy': 'E3_PASS iff GD_flip and GD_dose and GD_spec; '
                        'GD_amp reported as amplitude-carrier support; '
                        'partial => E3_DIRECTIONAL; else FAIL (FAIL goes '
                        'to F ledger, not hidden)',
}

# ---------- STEP 0: device facts (token ids, before freeze) ----------
import glob
from tokenizers import Tokenizer
HUB = os.path.join(os.path.expanduser('~'), '.cache', 'huggingface', 'hub')
SNAP = {'qwen3-4b': glob.glob(os.path.join(HUB, 'models--Qwen--Qwen3-4B-Base', 'snapshots', '*'))[0],
        'qwen3-14b': glob.glob(os.path.join(HUB, 'models--Qwen--Qwen3-14B', 'snapshots', '*'))[0]}
# tokenizer provenance per C9: 4B caches were collected with the LOCAL model
TOKPATH = {'qwen3-4b': os.path.join(ROOT, 'models', 'hf', 'qwen3-4b', 'tokenizer.json'),
           'qwen3-14b': os.path.join(SNAP['qwen3-14b'], 'tokenizer.json')}
TOK = {}
for m in MODELS:
    tk = Tokenizer.from_file(TOKPATH[m])
    TOK[m] = tk
    assert tk.encode('是一种', add_special_tokens=False).ids == [TOK_ISA], (m, 'isa token')
    assert tk.encode('有', add_special_tokens=False).ids == [TOK_HASA], (m, 'hasa token')
    assert tk.encode('是', add_special_tokens=False).ids == [TOK_SHI], (m, 'shi token')
    assert TOK_ISA != TOK_HASA
    cls_tok = [tk.encode(c, add_special_tokens=False).ids[0] for c in CLS_NAMES]
    assert len(set(cls_tok)) == 10, ('D2 distinctness replication', m, cls_tok)
    log('STEP0 %s tokens OK: isa=%d hasa=%d shi=%d CLS_TOK distinct x10 (D2 replication)'
        % (m, TOK_ISA, TOK_HASA, TOK_SHI))
CLS_TOK = {m: [TOK[m].encode(c, add_special_tokens=False).ids[0] for c in CLS_NAMES]
           for m in MODELS}

# ---------- STEP 1: FREEZE (before any observation) ----------
fz = {
    'phase': '3228', 'name': NAME,
    'prereg': 'AGI_GPT5_MEMO Phase 3227 tail (frozen before any observation; '
              'default arm (a) compute+seal, zero GPU compute)',
    'design_basis': {'design_doc': 'e3_intervention_design_v1.json',
                     'design_content_sha8': 'a883c4da', 'design_file_sha8': '32968738',
                     'design_freeze_sha8': '82dbb8b1'},
    'models': MODELS,
    'panels': {
        'seen_3154': {'file': r'phase3154\g1p4_mfd_multifactor_disentangle\{m}\collect.npz',
                      'sha8': CACHE_3154, 'rows': 208,
                      'final_slot': 'index NH-1 (= NL): 36 (4b) / 40 (14b) — '
                                    'the POST-NORM pre-logit state [C9]',
                      'folds': 'ci in {0,1,2,3} over train rows (ho=0), 48 rows '
                               'each [C3]; heldout ho=1 sealed descriptive extra',
                      'tags': ['g', 'l', 's', 'ti', 'ci', 'ho']},
        'oov_3169': {'file': r'phase3169\g5a6_oov_panel\collect_{m}.npz',
                     'sha8': CACHE_3169, 'rows': 2190,
                     'final_slot': 'index NH-1: 36 (4b) / 40 (14b) — post-norm '
                                   'pre-logit state [C9]',
                     'slices': 'template t in {0,1,2} [C1: first dim is 3 '
                               'TEMPLATES, design-doc "3 seeds" corrected]',
                     'tags': ['t', 'pi', 'pair(i,c)', 'is_oov_class', 'oov_class']},
    },
    'judgment': {
        'formula': 'J = sign(mean-logit isa-token-set - mean-logit '
                   'hasa-token-set) on exact post-patch logits [3227 D3]',
        'isa_set': [TOK_ISA], 'hasa_set': [TOK_HASA],
        'token_provenance': '是一种=single token 101158 (sealed 3169 TPL0/TPL2 '
                            '+ 3157 isa template); 有=single token 18830 (3157 '
                            'hasa template); ids verified on both tokenizers [C2]',
        'distinctness': '101158 != 18830 asserted; 3169 D2 CLS_TOK x10 '
                        'distinctness re-verified per model',
        'char_pair_descriptive': [TOK_SHI, TOK_YOU],
        'flip': 'flip(h, R, a) = [J_before = not R] and [J_after = R] '
                '(3227 math verbatim); ties asserted zero [C6]',
    },
    'directions': {
        'u_R': 'T2 chain recipe verbatim: 3157 cache, dC = H[ctx1]-H[ctx0] at '
               'KOUT=NL-1, dCn row-normalized, u_R = unit_mean(dCn[rel=R]) '
               '(3177-3187 chain; 3183 cos_rl/norm_dv anchors <1e-9)',
        'ustar': 'sealed 3225 npz a1_behavioral_band.npz (ac5d4d74), key {m}_ustar',
        'injection_site': 'final pre-logit (post-norm) state, cached slot '
                          'NH-1 of 3154/3169 rows [C9]; kout-layer directions '
                          'injected at the pre-logit state IS the behavioral '
                          'test (3225 stance, logged)',
    },
    'weights': {
        'map': 'l = W_U @ (h + dv), h = cached pre-logit state; readout map '
               'EXACTLY LINEAR — the design ln_f factor is already absorbed '
               'in the cached slot NH-1 [C9]',
        'sources': {m: dict(path=WU_SOURCES[m]['path'].replace(ROOT, ''),
                            key=WU_SOURCES[m]['key'], sha8=WU_SOURCES[m]['sha8'],
                            provenance=WU_SOURCES[m]['provenance'])
                    for m in MODELS},
        'cross_phase_note': '3225 A1 used the hub 4B W_U (4c807e25) on '
                            'local-collected residuals; A1 linearity '
                            'conclusion unaffected (map property), '
                            'provenance caveat recorded for 3229 [C9]',
    },
    'scale': {'alphas': list(ALPHAS),
              'dv': 'unit(u) * alpha * rms(h_row), h_row = cached pre-logit '
                    'state (post-norm; RMS ~2.2) [C4 + C9]',
              'nulls': {'random': '10 pinned seeds %s, unit direction per '
                                  'seed (per-model D), norm-matched to dv '
                                  'per row' % NULL_SEEDS,
                        'orth': 'v = unit(h_row - (h_row . u_R) u_R), '
                                'norm-matched to dv [C5]'},
              },
    'gd_amp_pinning': {'collapse_c': 'mean over class-c OOV rows x seeds '
                                     '7/8/9 of 3169-B normalized MSE (per '
                                     'seed class-mean, then mean over seeds)',
                       'amplitude_c': 'median |Qn_row . u*| over class-c rows '
                                      '(Qn = 3169 B-protocol row directions, '
                                      '3194 verbatim)',
                       'gate_stat': 'Spearman(amplitude_c, collapse_c) >= 0.6, '
                                    'positive sign pinned [C7]'},
    'device_gates': {
        'd_3157_tc': 'tc_mean_cos anchor <1e-9 per model',
        'd_3183_dv': 'cos_rl/norm_dv replication <1e-9 per model',
        'd_3169_E': 'per-seed E_seen/E_oov replication <1e-9 per model',
        'd_3189_90_91': '3189/3190/3191 scalars == 0 bitwise per model',
        'd_ustar': 'recomputed u* == sealed 3225 npz u* (bitwise)',
        'd_marg_c6v3': '[C6v3] baseline direct-map class readout (W_U @ '
                       'h_prelogit) vs sealed 3169 bf16 marg: argmax '
                       'agreement >= 0.98 per model (arbitrated values: '
                       '0.9963 local-4B / 0.9941 14B; centered-cos 1.0000 '
                       'both; supersedes C6 v1 0.90-blanket and C6v2 0.80 '
                       'floor, both computed under the superseded '
                       'double-norm map)',
    },
    'gates_3227_verbatim': GATES_3227_VERBATIM,
    'gate_application': 'DEFERRED to 3229 (registry decision arm); this arm '
                        'seals matrices + nulls + amplitude correlations only',
    'e3_taxonomy': 'in-protocol causality only — last-layer residual '
                   'injection bypasses upstream computation; no '
                   'full-regeneration claim (3227 design verbatim)',
    'zero_gpu': 'CPU matrix math only; no model forward, no GPU',
}
fz['sha8'] = csha8(dict(fz))
with open(os.path.join(OUTDIR, PFX + 'execution.json'), 'w',
          encoding='utf-8', newline='') as f:
    json.dump(fz, f, ensure_ascii=False, indent=1)
log('== Phase 3228 %s %s == freeze %s (before any observation)'
    % (NAME, 'SMOKE' if SMOKE else 'FULL', fz['sha8']))

# ---------- sealed inputs ----------
sealed_3169 = json.load(open(p94.SRC['p3169_result'], encoding='utf-8'))
sealed_3183 = json.load(open(p94.SRC['p3183_result'], encoding='utf-8'))
sealed_3189 = json.load(open(p94.SRC['p3189_result'], encoding='utf-8'))
sealed_3190 = json.load(open(p94.SRC['p3190_result'], encoding='utf-8'))
sealed_3191 = json.load(open(p94.SRC['p3191_result'], encoding='utf-8'))
assert sealed_3183['res_sha8'] == 'c55e4e5e' and sealed_3183['seal_sha8'] == '7196fa8a'
NPZ25 = os.path.join(SRC_DIR, 'phase3225', 'g5g25_a1_behavioral_band_compute',
                     'a1_behavioral_band.npz')
assert sha8_file(NPZ25) == 'ac5d4d74'
CFG = {m: json.load(open(os.path.join(SNAP[m], 'config.json'), encoding='utf-8'))
       for m in MODELS}
EPS = {m: float(CFG[m].get('rms_norm_eps', 1e-6)) for m in MODELS}


def spearman(x, y):
    rx = np.argsort(np.argsort(np.asarray(x, dtype=np.float64))).astype(np.float64)
    ry = np.argsort(np.argsort(np.asarray(y, dtype=np.float64))).astype(np.float64)
    return float(np.corrcoef(rx, ry)[0, 1])


PANEL_3154 = {
    'qwen3-4b': os.path.join(SRC_DIR, r'phase3154\g1p4_mfd_multifactor_disentangle\qwen3-4b\collect.npz'),
    'qwen3-14b': os.path.join(SRC_DIR, r'phase3154\g1p4_mfd_multifactor_disentangle\qwen3-14b\collect.npz'),
}
PANEL_3169 = {m: p94.SRC[p94.NPZ69_KEY[m]] for m in MODELS}

out = {}
npz_payload = {}
for m in MODELS:
    log('=== %s ===' % m)
    D = int(CFG[m]['hidden_size'])
    NL = int(CFG[m]['num_hidden_layers'])
    NH = NL + 1
    fin = NH - 1
    eps = EPS[m]

    # --- u_R directions (T2 recipe verbatim, 3157 cache) ---
    A57 = p94.ANCH_3157[p94.ANCH_KEY[m]]
    z57 = np.load(p94.SRC[p94.NPZ57_KEY[m]])
    H57 = z57['H'].astype(np.float32)
    assert H57.shape == (128, NH, D), ('3157 shape', m, H57.shape)
    EI = z57['ei'].astype(int)
    RL = z57['rel'].astype(int)
    PL = z57['pol'].astype(int)
    CX = z57['ctx'].astype(int)
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
    v_isa = p94.unit_mean(dCn[RL_rows[:, 1] == 0])
    v_hasa = p94.unit_mean(dCn[RL_rows[:, 1] == 1])
    cos_rl = float(v_isa @ v_hasa)
    norm_dv = float(np.linalg.norm(v_isa - v_hasa))
    E83 = sealed_3183['per_model'][m]
    d_83 = max(abs(cos_rl - E83['cos_rl']), abs(norm_dv - E83['norm_dv']))
    assert d_83 < 1e-9, ('3183 dv anchor', m, d_83)
    log('  u_R anchors OK: tc drift %.2e, 3183 drift %.2e (cos_rl=%.6f norm_dv=%.6f)'
        % (d_tc, d_83, cos_rl, norm_dv))
    del z57, H57, dC, dCn, cosM

    # --- u* from sealed 3225 npz ---
    z25 = np.load(NPZ25)
    ustar = z25[m + '_ustar'].astype(np.float32)
    assert ustar.shape == (D,)
    del z25

    # --- Qn pipeline (3169 B-protocol verbatim) + per-class data ---
    rd = int(sealed_3169['per_model'][m]['readout'])
    assert rd == NL - 1, ('readout slot', m)
    z69 = np.load(PANEL_3169[m])
    H69 = z69['H']
    MARG69 = z69['marg']
    assert H69.shape == (3, 730, NH, D), ('3169 shape', m, H69.shape)
    Y = H69[:, :, rd, :].reshape(3 * 730, D).astype(np.float32)
    H69_final = H69[:, :, fin, :].reshape(3 * 730, D).astype(np.float32)
    tr_base = {}
    for s in p94.SEEDS_S1:
        tr_pairs, te_pairs = p94.split_s1_246(s)
        tr_base[s] = [t * 730 + p94.PI_OF_PAIR[p] for t in range(3)
                      for p in p94.PAIRS if p in tr_pairs]
    oov_rows = p94.ALL_OOV_ROWS
    oov_row_cls = np.array([p94.PAIRS[r % 730][1] for r in oov_rows])

    def rv(r, ne):
        i, c = p94.PAIRS[r % 730]
        return p94.phi_row(i, c, r // 730, ne, p94.NC, p94.NT)

    E69 = sealed_3169['per_model'][m]['B']
    drift69 = 0.0
    Q_rows = np.zeros((len(oov_rows), D), dtype=np.float32)
    E_cls_seed = np.zeros((3, 4), dtype=np.float64)  # per seed x per OOV class
    for si, s in enumerate(p94.SEEDS_S1):
        tr_rows = tr_base[s]
        Xtr = np.stack([rv(r, p94.NE) for r in tr_rows])
        W = p94.ridge_primal(Xtr, Y[tr_rows])
        ref = Y[tr_rows].mean(0)
        Dk = float(((Y[tr_rows] - ref) ** 2).sum(1).mean()) + 1e-9
        te_pairs = p94.split_s1_246(s)[1]
        te_rows = [t * 730 + p94.PI_OF_PAIR[p] for t in range(3)
                   for p in p94.PAIRS if p in te_pairs]
        Xs = np.stack([rv(r, p94.NE) for r in te_rows])
        es = ((Xs @ W - Y[te_rows]) ** 2).sum(1) / Dk
        Xo = np.stack([rv(r, p94.NE) for r in oov_rows])
        eo = ((Xo @ W - Y[oov_rows]) ** 2).sum(1) / Dk
        drift69 = max(drift69, abs(float(es.mean()) - E69['E_seen_per_seed'][si]),
                      abs(float(eo.mean()) - E69['E_oov_per_seed'][si]))
        for ci_, c in enumerate(range(6, 10)):
            E_cls_seed[si, ci_] = float(eo[oov_row_cls == c].mean())
        R_oov = Y[oov_rows] - Xo @ W
        for j in range(len(oov_rows)):
            rvv = R_oov[j]
            Q_rows[j] += rvv / (float(np.linalg.norm(rvv)) + 1e-18)
        del W, Xtr
    assert drift69 < 1e-9, ('3169 per-seed E replication', m, drift69)
    Qn = Q_rows / len(p94.SEEDS_S1)
    Qn = Qn / (np.linalg.norm(Qn, axis=1, keepdims=True) + 1e-18)
    log('  Qn pipeline OK: drift69=%.2e' % drift69)

    # --- 3189/3190/3191 anchors + ustar bitwise ---
    s0 = np.linalg.svd(Qn, compute_uv=False)
    energy = s0 ** 2
    tot = float(energy.sum())
    top1 = float(energy[0] / tot)
    cls_ids = [6, 7, 8, 9]
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
    E89 = sealed_3189['per_model'][m]
    drift89 = max(abs(top1 - E89['top1']),
                  max(abs(a - b) for a, b in zip(within, E89['within_class'])),
                  abs(cross_med - E89['cross_med']))
    assert drift89 == 0.0, ('3189 scalar drift', m, drift89)
    U, S, Vt = np.linalg.svd(Qn, full_matrices=False)
    ustar_rec = Vt[0]
    d_ustar = float(np.abs(ustar_rec - ustar).max())
    assert d_ustar == 0.0, ('u* bitwise vs sealed 3225 npz', m, d_ustar)
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
                  max(abs(a - b) for a, b in zip(med_abs, E91['med_abs_cos_held'])))
    assert drift91 == 0.0, ('3191 control-anchor drift', m, drift91)
    log('  device anchors OK: drift89=0 drift90=0 drift91=0 u*_bitwise=0')

    # --- GD_amp data: per-class collapse + amplitude + spearman ---
    E_cls = E_cls_seed.mean(axis=0)  # (4,) mean over seeds
    amp_med = []
    amp_mean = []
    amp_signed_med = []
    for c in cls_ids:
        idxs = np.where(oov_row_cls == c)[0]
        proj = Qn[idxs] @ ustar
        amp_med.append(float(np.median(np.abs(proj))))
        amp_mean.append(float(np.abs(proj).mean()))
        amp_signed_med.append(float(np.median(proj)))
    rho_amp = spearman(amp_med, E_cls)
    log('  GD_amp data: E_cls %s | amp_med %s | spearman=%.4f'
        % ('/'.join('%.4f' % x for x in E_cls),
           '/'.join('%.4f' % x for x in amp_med), rho_amp))

    # --- weights (C9 provenance-corrected sources) ---
    from safetensors import safe_open
    src = WU_SOURCES[m]
    assert sha8_file(src['path']) == src['sha8'], ('W_U sha', m, src['path'])
    with safe_open(src['path'], framework='pt') as f:
        avail = list(f.keys())
        assert src['key'] in avail, ('key missing', m, src['key'])
        W_U = f.get_tensor(src['key']).float().numpy()
    wu_key = src['key']
    assert W_U.shape == (int(CFG[m]['vocab_size']), D), ('W_U shape', m, W_U.shape)
    WT = W_U[[TOK_ISA, TOK_HASA]].astype(np.float32)        # (2, D) primary sets
    WT_char = W_U[[TOK_SHI, TOK_YOU]].astype(np.float32)    # descriptive pair
    WUC = W_U[CLS_TOK[m]].astype(np.float32)                # (10, D)
    cos_uR_u = [float(v_isa @ ustar), float(v_hasa @ ustar)]
    log('  weights [C9]: %s %s (%s) | cos(u_isa,u*)=%.4f cos(u_hasa,u*)=%.4f'
        % (wu_key, W_U.shape, src['provenance'][:40], cos_uR_u[0], cos_uR_u[1]))
    del W_U

    # ---------- flip matrices: seen panel (3154) ----------
    def flip_block(tag, Hrows, slices, slice_names, npz_prefix):
        """Hrows (n, D) fp32 final-layer; slices: list of int arrays (row idx
        per slice). Returns per-row margins dict + aggregates."""
        n = Hrows.shape[0]
        rms = np.sqrt((Hrows ** 2).mean(axis=1, keepdims=True))
        # direct linear readout map [C9]: l = W_U @ h, h = pre-logit state
        m0 = Hrows @ WT.T                     # (n, 2) primary margins
        m0c = Hrows @ WT_char.T               # (n, 2) char pair descriptive
        c0 = Hrows @ WUC.T                    # (n, 10) class readout
        J0 = (m0[:, 0] > m0[:, 1]).astype(np.int8)   # 1=isa-side, 0=hasa-side
        n_tie = int((m0[:, 0] == m0[:, 1]).sum())
        if n_tie:
            bad.append('tie baseline %s n=%d' % (tag, n_tie))
        # device anchor C6v3: baseline direct-map readout vs sealed marg (OOV)
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
                '(n=%d) gap>=1.0=%.4f (n=%d) centered-cos=%.4f'
                % (tag, marg_diag['agree_all'], marg_diag['agree_gap05'],
                   marg_diag['n_gap05'], marg_diag['agree_gap10'],
                   marg_diag['n_gap10'], gm))
            if marg_diag['agree_all'] < 0.98:
                bad.append('C6v3 floor %s=%.4f' % (m, marg_diag['agree_all']))
        # perturbations
        perts = [('u_isa', v_isa[None, :] * np.ones((n, 1), np.float32), 'real'),
                 ('u_hasa', v_hasa[None, :] * np.ones((n, 1), np.float32), 'real')]
        for s in NULL_SEEDS:
            rng = np.random.RandomState(s)
            v = rng.randn(D).astype(np.float32)
            v /= float(np.linalg.norm(v))
            perts.append(('rand_%d' % s, v[None, :] * np.ones((n, 1), np.float32), 'null'))
        for rname, u in (('orth_isa', v_isa), ('orth_hasa', v_hasa)):
            unh = u / (float(np.linalg.norm(u)) + 1e-18)
            proj = (Hrows @ unh)[:, None] * unh[None, :]
            wo = Hrows - proj
            wo = wo / (np.linalg.norm(wo, axis=1, keepdims=True) + 1e-18)
            perts.append((rname, wo, 'null'))
        margins = np.zeros((len(perts), len(ALPHAS), n, 2), np.float32)
        margins_char = np.zeros((len(perts), len(ALPHAS), n, 2), np.float32)
        class_logits = np.zeros((3, len(ALPHAS), n, 10), np.float32)
        for pi_, (pname, vunit, kind) in enumerate(perts):
            for ai_, a in enumerate(ALPHAS):
                dv = vunit * (a * rms)
                Hp = Hrows + dv
                margins[pi_, ai_] = Hp @ WT.T
                margins_char[pi_, ai_] = Hp @ WT_char.T
                if kind == 'real':
                    class_logits[pi_, ai_] = Hp @ WUC.T
                del Hp, dv
        # flip aggregation: flip toward R = [J0 == not R] and [J_after == R]
        agg = {}
        for si_, sl in enumerate(slices):
            for R, Rname in ((1, 'isa'), (0, 'hasa')):
                # R encoding: J=1 isa-side; target isa (R=1): before must be 0
                for pi_, (pname, _, kind) in enumerate(perts):
                    for ai_, a in enumerate(ALPHAS):
                        J1 = (margins[pi_, ai_, :, 0] > margins[pi_, ai_, :, 1])
                        fl = (J0[sl] == (1 - R)) & (J1[sl].astype(np.int8) == R)
                        nr = int(len(sl))
                        nf = int(fl.sum())
                        agg['%s|%s|%s|a%s' % (Rname, pname, slice_names[si_], a)] = \
                            dict(n=nf, rows=nr, rate=float(nf / nr))
        npz_payload[npz_prefix + '_margins'] = margins
        npz_payload[npz_prefix + '_margins_char'] = margins_char
        npz_payload[npz_prefix + '_class_logits'] = class_logits
        npz_payload[npz_prefix + '_J0'] = J0
        npz_payload[npz_prefix + '_m0'] = m0
        # dose table: mean margin-delta dD = d(after) - d(before), d = m_isa - m_hasa
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

    # seen panel
    z54 = np.load(PANEL_3154[m])
    H54 = z54['H']
    assert H54.shape == (208, NH, D), ('3154 shape', m, H54.shape)
    tags54 = {k: z54[k].astype(int) for k in ('g', 'l', 's', 'ti', 'ci', 'ho')}
    del z54
    H54f = H54[:, fin, :].astype(np.float32)
    del H54
    sls54 = [np.where((tags54['ci'] == c) & (tags54['ho'] == 0))[0] for c in range(4)]
    assert all(len(s) == 48 for s in sls54), ('fold sizes', [len(s) for s in sls54])
    sls54_desc = np.where(tags54['ho'] == 1)[0]
    seen_res = flip_block('seen', H54f, sls54, ['cut%d' % c for c in range(4)], m + '_seen')
    seen_res['heldout_rows'] = int(len(sls54_desc))
    # heldout descriptive aggregate (alpha=1, real perts)
    ho_desc = {}
    for R, Rname in ((1, 'isa'), (0, 'hasa')):
        for pname in ('u_isa', 'u_hasa'):
            pi_ = seen_res['perts'].index(pname)
            ai_ = ALPHAS.index(1.0)
            J1 = (npz_payload[m + '_seen_margins'][pi_, ai_, :, 0]
                  > npz_payload[m + '_seen_margins'][pi_, ai_, :, 1])
            fl = (npz_payload[m + '_seen_J0'][sls54_desc] == (1 - R)) & \
                 (J1[sls54_desc].astype(np.int8) == R)
            ho_desc['%s|%s' % (Rname, pname)] = \
                dict(n=int(fl.sum()), rows=int(len(sls54_desc)),
                     rate=float(fl.sum() / len(sls54_desc)))
    seen_res['heldout_desc_a1'] = ho_desc
    del H54f

    # OOV panel
    sls69 = [np.arange(t * 730, (t + 1) * 730) for t in range(3)]
    oov_res = flip_block('oov', H69_final, sls69, ['tpl%d' % t for t in range(3)],
                         m + '_oov')
    oov_res['row_tags'] = {'oov_class_rows': int(len(oov_rows)),
                           'seen_class_rows': int(2190 - len(oov_rows))}
    # OOV-class-only descriptive aggregate (rows in templates t, pair c>=6)
    oov_row_mask = np.zeros(2190, bool)
    for t in range(3):
        for r in oov_rows:
            oov_row_mask[t * 730 + (r % 730)] = True
    oov_cls_desc = {}
    for R, Rname in ((1, 'isa'), (0, 'hasa')):
        for pname in ('u_isa', 'u_hasa'):
            pi_ = oov_res['perts'].index(pname)
            for ai_, a in enumerate(ALPHAS):
                J1 = (npz_payload[m + '_oov_margins'][pi_, ai_, :, 0]
                      > npz_payload[m + '_oov_margins'][pi_, ai_, :, 1])
                fl = (npz_payload[m + '_oov_J0'][oov_row_mask] == (1 - R)) & \
                     (J1[oov_row_mask].astype(np.int8) == R)
                oov_cls_desc['%s|%s|a%s' % (Rname, pname, a)] = \
                    dict(n=int(fl.sum()), rows=int(oov_row_mask.sum()),
                         rate=float(fl.sum() / oov_row_mask.sum()))
    oov_res['oov_class_desc'] = oov_cls_desc
    del z69, H69, H69_final, Y, Q_rows, Qn, MARG69

    # summary numbers for the log (matched vs null at alpha=1, seen folds)
    for Rname in ('isa', 'hasa'):
        for f in ('cut0', 'cut1', 'cut2', 'cut3'):
            km = '%s|matched_dir|a1.0' % Rname
            matched = seen_res['agg']['%s|u_%s|%s|a1.0' % (Rname, Rname, f)]
            nullmax = max(vv['rate'] for kk, vv in seen_res['agg'].items()
                          if kk.endswith('|%s|a1.0' % f)
                          and kk.split('|')[1].startswith(('rand_', 'orth_')))
            log('  seen %s %s: matched=%.4f nullmax=%.4f diff=%+.4f'
                % (Rname, f, matched['rate'], nullmax, matched['rate'] - nullmax))

    log('  dose (seen): %s' % '; '.join(
        '%s a%s dD=%+.4f' % (pn, a, seen_res['dose']['%s|a%s' % (pn, a)])
        for pn in ('u_isa', 'u_hasa') for a in ('0.5', '1.0', '2.0')))
    log('  dose (oov): %s' % '; '.join(
        '%s a%s dD=%+.4f' % (pn, a, oov_res['dose']['%s|a%s' % (pn, a)])
        for pn in ('u_isa', 'u_hasa') for a in ('0.5', '1.0', '2.0')))

    out[m] = dict(
        u_R_anchors=dict(tc_drift=d_tc, dv_drift=d_83, cos_rl=cos_rl,
                         norm_dv=norm_dv),
        device=dict(drift69=drift69, drift89=drift89, drift90=drift90,
                drift91=drift91, ustar_bitwise=d_ustar,
                marg_diag=oov_res['marg_diag']),
        gd_amp=dict(E_cls=[float(x) for x in E_cls],
                    E_cls_seed=[[float(x) for x in row] for row in E_cls_seed],
                    amp_med=amp_med, amp_mean=amp_mean,
                    amp_signed_med=amp_signed_med,
                    spearman_amp_collapse=rho_amp),
        geometry=dict(cos_u_isa_ustar=cos_uR_u[0], cos_u_hasa_ustar=cos_uR_u[1],
                      cos_isa_hasa=cos_rl),
        seen=seen_res, oov=oov_res,
        wu_key=wu_key, eps=eps, final_layer=fin,
    )
    log('  %s done (%.1fs)' % (m, time.time() - T0))

# ---------- STEP 3: SEAL ----------
npz_path = os.path.join(OUTDIR, PFX + 'e3_flip_matrices.npz')
np.savez_compressed(npz_path, **npz_payload)
clean = not bad
verdict = 'e3_compute_sealed' if clean else 'e3_compute_sealed_dirty(%d)' % len(bad)
led = json.load(open(os.path.join(ROOT, 'research', 'gpt5', 'atlas',
                                  'atlas_ledger.json'), encoding='utf-8'))
assert len(led['measurements']) == 379 and led['ledger_sha256_8'] == 'f21108e7'
log('ledger untouched: n=379 post=f21108e7 before==after')

res = {
    'phase': 3228, 'name': NAME,
    'arm': 'compute+seal (default (a), zero GPU compute, CPU matrix math only)',
    'prereg_authorization_record': '"好的，继续" = preregistered-default-arm trigger',
    'freeze_sha8': fz['sha8'],
    'design_basis': fz['design_basis'],
    'clarifications': ['C1 3169 first dim = 3 templates (doc said 3 seeds)',
                       'C2 isa-set {101158} hasa-set {18830} from sealed '
                       'prompt ids; D2 distinctness replicated',
                       'C3 seen folds = comma-cut groups ci {0..3} (ho=0)',
                       'C4 dv = unit(u)*alpha*rms(h_row) (A1 convention, '
                       'h_row = pre-logit state)',
                       'C5 orth null = unit(h - (h.u)u) norm-matched',
                       'C9 readout map corrected: cached slot NH-1 IS the '
                       'post-norm pre-logit state (HF final-norm append); '
                       'map exactly linear l = W_U @ (h + dv); 4B W_U '
                       'provenance split found and corrected to LOCAL '
                       'collector weights (328a91d3); 14B hub==local '
                       '(0d6b9229); cross-phase 3225 provenance note '
                       'recorded for 3229',
                       'C6v3 device gate: baseline direct-map marg argmax '
                       'agreement >= 0.98 (arbitrated: 0.9963/0.9941, '
                       'centered-cos 1.0000; supersedes C6 v1/v2 computed '
                       'under the superseded double-norm map)',
                       'C7 GD_amp stat = Spearman(med|Qn.u*|_c, E_c) >= 0.6',
                       'C8 SMOKE finding under the superseded double-norm '
                       'map: matched flips 0; u_isa anti-directional '
                       'monotone; corrected map re-measures'],
    'models': {m: dict(
        u_R_anchors=out[m]['u_R_anchors'],
        device=out[m]['device'],
        gd_amp=out[m]['gd_amp'],
        geometry=out[m]['geometry'],
        seen=dict(n_rows=out[m]['seen']['n_rows'],
                  J0_isa_share=out[m]['seen']['J0_isa_share'],
                  d0_mean=out[m]['seen']['d0_mean'],
                  d0_std=out[m]['seen']['d0_std'],
                  marg_diag=out[m]['seen']['marg_diag'],
                  dose=out[m]['seen']['dose'],
                  agg=out[m]['seen']['agg'],
                  heldout_desc_a1=out[m]['seen']['heldout_desc_a1']),
        oov=dict(n_rows=out[m]['oov']['n_rows'],
                 J0_isa_share=out[m]['oov']['J0_isa_share'],
                 d0_mean=out[m]['oov']['d0_mean'],
                 d0_std=out[m]['oov']['d0_std'],
                 marg_diag=out[m]['oov']['marg_diag'],
                 dose=out[m]['oov']['dose'],
                 agg=out[m]['oov']['agg'],
                 oov_class_desc=out[m]['oov']['oov_class_desc']),
    ) for m in MODELS},
    'sealed_npz': {'file': PFX + 'e3_flip_matrices.npz',
                   'file_sha8': sha8_file(npz_path)},
    'gate_application': 'DEFERRED to 3229 per design chain (a883c4da)',
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
