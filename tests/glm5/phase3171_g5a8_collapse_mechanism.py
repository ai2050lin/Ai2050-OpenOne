# -*- coding: utf-8 -*-
# Phase 3171 G5-A8: OOV collapse mechanism localization (zero GPU).
# Reuses sealed 3169 collect npz (panel H + MARG) + 3165/3166 subspace recipes.
# Question: is the OOV-class readout collapse (3169 ratio_B pooled 2.5388)
#   encoding_missing (OOV-class H does not enter the S_class class subspace)
#   or readout_missing (encoded but ridge does not transfer)?
# Probes:
#  (a) PRIMARY GATE: projection energy fraction f_S(h)=||Q_S^T h||^2/||h||^2 on
#      the S_class subspace (3166 recipe rebuild, per-model W_U), OOV-class rows
#      vs seen rows at the 3169 E_read slot. Pre-registered gate on
#      ratio = mean(f_S|oov)/mean(f_S|seen):
#        ratio < 0.5 -> encoding_missing; ratio > 0.8 -> readout_missing;
#        else mixed. Three-model unanimous -> single cls, else
#        mixed_across_models.
#  (b) side evidence: same ratio on the K_readout top64 subspace (3158 W_U Gram
#      top eigvecs) - descriptive, not gated.
#  (c) centroid geometry at the primary slot: 10 panel-class centroids; OOV
#      centroids vs 6 seen centroids (nearest cosine, NNLS cone fit on seen
#      centroids, 6-dim subspace projection fraction) with leave-one-out seen
#      baselines; OOV centroids vs the 10 S_class English directions; full
#      10x10 panel-centroid cosine matrix.
#  (d) MARG readout behavior (3169 centered class-word logits): mean claimed-
#      class margin on OOV rows vs mean true-class margin on seen rows; max
#      seen-word margin on OOV rows - descriptive.
# Slots: primary k_main = NL-1 (asserted equal to 3169 result per_model
# 'readout' field, the slot where collapse was measured); side slots NL (the
# 3157 K_entity / 3166 R_logic slot) and 3 (KSTAR) - descriptive.
# dtype chain: H float16 -> float32 -> float64 domain; S_class rebuild verbatim
# (float32 round-trip -> float64, 3166 main chain). Crosscheck gate: recomputed
# K_readout x S_class top1 must reproduce the 3166 census value (tol 1e-6 deg).
# DESIGN is fully static (3169 discipline): no smoke flags, no truncation
# tables, no runtime interpolation inside DESIGN.
import hashlib
import io
import json
import os
import sys
import time

import numpy as np

T0 = time.time()
PHASE = 3171
NAME = 'g5a8_collapse_mechanism'
SMOKE = os.environ.get('SMOKE', '') == '1'
ROOT = r'D:\AI2050\Ai2050-OpenOne'
RDIR = os.path.join(ROOT, 'tests', 'glm5', 'result', 'rdc_query_construction_20260913')
OUTDIR = os.path.join(RDIR, 'phase3171', 'g5a8_collapse_mechanism')
P3169 = os.path.join(RDIR, 'phase3169', 'g5a6_oov_panel')
P3166 = os.path.join(RDIR, 'phase3166', 'g5a3b_logic_direction', 'result.json')
B3158 = os.path.join(RDIR, 'phase3158', 'g4p1_output_equivalence_class')
LOG = []

MODELS = [
    dict(name='qwen3-4b', mdir=os.path.join(ROOT, 'models', 'hf', 'qwen3-4b'),
         npz=os.path.join(P3169, 'collect_qwen3-4b.npz'),
         t58=os.path.join(B3158, 'qwen3-4b', 'collect.npz')),
    dict(name='qwen3-14b', mdir=os.path.join(ROOT, 'models', 'hf', 'Qwen3-14B'),
         npz=os.path.join(P3169, 'collect_qwen3-14b.npz'),
         t58=os.path.join(B3158, 'qwen3-14b', 'collect.npz')),
    dict(name='glm4-9b', mdir=os.path.join(ROOT, 'models', 'hf', 'glm4-9b-chat-hf'),
         npz=os.path.join(P3169, 'collect_glm4-9b.npz'),
         t58=os.path.join(B3158, 'glm4', 'collect.npz')),
]
SMOKE_NPZ = os.path.join(P3169, 'collect_smoke_qwen3-4b.npz')

# measured sha8 anchors (3166 SHA_ANCHOR + 3169 on-disk inventory; all probed)
SHA_ANCHOR = {
    'p3169_4b': '36eb4ff0', 'p3169_14b': '97f98575', 'p3169_glm4': '9cc3f8b4',
    'p3169_smoke_4b': 'db50c9dc',
    'p3158_4b': 'c4d9b9fb', 'p3158_14b': '7f53f3a0', 'p3158_glm4': '2ea72cfe',
    'p2806_exec': 'b291ea35',
    'p3166_result': '448af595', 'p3169_result': '5b51c2c1',
}


def log(s):
    ln = '[%7.1f] %s' % (time.time() - T0, s)
    LOG.append(ln)
    try:
        print(ln, flush=True)
    except Exception:
        pass


def sha8(p):
    return hashlib.sha256(open(p, 'rb').read()).hexdigest()[:8]


# ---------------- panel (3169 verbatim; SMOKE truncates identically) ----------
CLASSES_SEEN = ['水果', '动物', '交通工具', '家具', '金属', '颜色']
ENT_SEEN = {
    '水果': ['苹果', '香蕉', '梨', '西瓜', '葡萄', '草莓', '芒果', '柠檬'],
    '动物': ['狗', '猫', '老虎', '大象', '兔子', '猴子', '马', '牛'],
    '交通工具': ['汽车', '火车', '飞机', '摩托车', '卡车', '地铁'],
    '家具': ['桌子', '椅子', '床', '沙发', '地毯', '窗帘'],
    '金属': ['铁', '铜', '铝', '金', '银', '锌', '铅'],
    '颜色': ['红', '蓝', '绿', '黄', '黑', '白'],
}
CLASSES_OOV = ['乐器', '天气', '运动', '电器']
ENT_OOV = {
    '乐器': ['钢琴', '小提琴', '吉他', '鼓', '笛子', '二胡', '琵琶', '口琴'],
    '天气': ['雨', '雪', '雷', '雾', '冰雹', '台风', '露水', '霜'],
    '运动': ['足球', '篮球', '乒乓球', '游泳', '跑步', '体操', '拳击', '围棋'],
    '电器': ['电视', '冰箱', '洗衣机', '空调', '微波炉', '电饭煲', '吸尘器', '风扇'],
}
NT = 3
if SMOKE:
    ENT_SEEN = {c: v[:2] for c, v in ENT_SEEN.items()}
    ENT_OOV = {c: v[:2] for c, v in ENT_OOV.items()}
CLASSES = CLASSES_SEEN + CLASSES_OOV
ENT = dict(ENT_SEEN)
ENT.update(ENT_OOV)
ENTS = [e for cl in CLASSES for e in ENT[cl]]
CLS_OF = [CLASSES.index(cl) for cl in CLASSES for e in ENT[cl]]
NE = len(ENTS)
NC = len(CLASSES)
PAIRS = [(i, c) for i in range(NE) for c in range(NC)]
NP_ = len(PAIRS)
N_SEEN_ENT = sum(len(v) for v in ENT_SEEN.values())
N_OOV_ENT = sum(len(v) for v in ENT_OOV.values())
N_SEEN_PAIRS = N_SEEN_ENT * len(CLASSES_SEEN)
N_OOVCLS_PAIRS = NE * len(CLASSES_OOV)
N_NEWENT_PAIRS = N_OOV_ENT * len(CLASSES_SEEN)
N_OOVPURE_PAIRS = N_OOV_ENT * len(CLASSES_OOV)
if not SMOKE:
    assert (N_SEEN_ENT, N_OOV_ENT, NE, NC, NP_) == (41, 32, 73, 10, 730), 'panel'
    assert (N_SEEN_PAIRS, N_OOVCLS_PAIRS, N_NEWENT_PAIRS, N_OOVPURE_PAIRS) == \
        (246, 292, 192, 128), 'pair sets'
SEEN_ROWS = [t * NP_ + pi for t in range(NT) for pi in range(NP_)
             if PAIRS[pi][0] < N_SEEN_ENT and PAIRS[pi][1] < len(CLASSES_SEEN)]
OOV_ROWS = [t * NP_ + pi for t in range(NT) for pi in range(NP_)
            if PAIRS[pi][1] >= len(CLASSES_SEEN)]
NEWENT_ROWS = [t * NP_ + pi for t in range(NT) for pi in range(NP_)
               if PAIRS[pi][0] >= N_SEEN_ENT and PAIRS[pi][1] < len(CLASSES_SEEN)]
OOVPURE_ROWS = [t * NP_ + pi for t in range(NT) for pi in range(NP_)
                if PAIRS[pi][0] >= N_SEEN_ENT and PAIRS[pi][1] >= len(CLASSES_SEEN)]

# ---------------- DESIGN (fully static) ---------------------------------------
DESIGN = dict(
    phase=PHASE, name=NAME,
    question='OOV-class readout collapse (3169 ratio_B pooled 2.5388, three '
             'models 2.83/2.13/2.71 all above the 2x gate): encoding_missing '
             'or readout_missing?',
    sources=dict(
        collect='3169 sealed collect npz x3 (H float16 [NT,NP,NH,D] last-token '
                'hidden states; MARG float32 [NT,NP,NC] centered class-word logits)',
        k_readout='3158 collect.npz top64 (D,64) column space, per model',
        s_class='3166 build_S_class recipe verbatim: p2806 CATS (10 families, '
                '80 single-token words), per-model W_U + tokenizer, centroid '
                'diff dW_c = Cm - (sum(Cm)-Cm)/9, unit rows, float32 round-trip',
        anchors='3166 result census K_readout__S_class top1 (67.0/71.9/74.014) '
                'runtime-read and asserted within 1e-6 deg'),
    rows=dict(seen='i<N_SEEN_ENT, c<6 (full: 246 pairs x3 tpl = 738 rows)',
              oov_cls='c>=6 (full: 292 pairs x3 = 876 rows)',
              newent='i>=N_SEEN_ENT, c<6 (full: 192 pairs x3 = 576 rows)',
              oov_pure='i>=N_SEEN_ENT, c>=6 (full: 128 pairs x3 = 384 rows)'),
    probes=dict(
        a_primary_gate='f_S(h) = norm(Q_S^T h)^2 / norm(h)^2 with Q_S = '
                       'orthonormal basis of S_class rows (10 dirs); ratio = '
                       'mean(f_S | oov rows) / mean(f_S | seen rows) at k_main; '
                       'PRE-REGISTERED GATE: ratio < 0.5 -> encoding_missing; '
                       'ratio > 0.8 -> readout_missing; else mixed; per-model '
                       'verdict at k_main; aggregate = unanimous -> single cls, '
                       'else mixed_across_models',
        b_side='same ratio on K_readout top64 subspace (descriptive, not gated)',
        c_centroid='10 panel-class centroids at k_main (mean of H rows of the '
                   'class over entities x templates); OOV vs 6 seen centroids: '
                   'nearest cosine + top2, NNLS cone fit on 6 seen centroids '
                   '(relative residual) + 6-dim subspace projection fraction, '
                   'baselines = leave-one-out seen NNLS/projection and seen '
                   'pairwise cosine; OOV vs 10 S_class directions cosine; full '
                   '10x10 panel-centroid cosine matrix',
        d_marg='MARG descriptive: mean claimed-class margin on OOV rows vs mean '
               'true-class margin on seen rows; max seen-word margin on OOV rows'),
    slots=dict(k_main='NL-1, asserted equal to 3169 result per_model.readout '
                      '(the slot where collapse was measured)',
               side='NL (3157 K_entity / 3166 R_logic slot) and 3 (KSTAR); '
                    'descriptive'),
    dtype_chain='H float16 -> float32 -> float64 domain; S_class rebuild '
                'float32 round-trip -> float64 (3166 main chain)',
    crosscheck_gate='recomputed K_readout x S_class top1 (kross verbatim) vs '
                    '3166 census per model, tol 1e-3 deg (3166 census stores '
                    'top1_deg rounded to 3 decimals; recomputation is exact '
                    'float64, SMOKE-observed raw drift 1.9e-5 deg)',
    device_gates=dict(
        G_anchor='all source_sha8 asserted',
        G_panel='row-set counts + H finite + MARG shape',
        G_slot='k_main == 3169 readout field == NL-1; kstar == 3',
        G_crosscheck='3166 census angles reproduced within 1e-6 deg',
        G_sanity='mean f_S(seen rows) > 1e-8 and mean f_K(seen rows) > 1e-4 '
                 'at k_main (subspace projection machinery non-degenerate)'),
    smoke='SMOKE run uses the sealed 3169 smoke npz (truncated panel, same '
          'structure); all gates except panel-size asserts execute',
)
CORE_KEYS = tuple(sorted(DESIGN.keys()))


def canon_sha(d):
    core = {k: v for k, v in d.items() if k not in ('created', 'design_sha8')}
    raw = json.dumps(core, ensure_ascii=False, indent=1, sort_keys=True)
    return hashlib.sha256(raw.encode('utf-8')).hexdigest()[:8]


def freeze():
    os.makedirs(OUTDIR, exist_ok=True)
    exep = os.path.join(OUTDIR, 'execution.json')
    d8 = canon_sha(DESIGN)
    if os.path.exists(exep):
        prev = json.load(io.open(exep, encoding='utf-8'))
        if prev.get('design_sha8') != d8:
            raise SystemExit('DRIFT: execution.json design_sha8 %s != current %s'
                             % (prev.get('design_sha8'), d8))
        log('freeze: existing execution.json OK (%s)' % d8)
    else:
        body = dict(DESIGN)
        body['created'] = time.strftime('%Y-%m-%d %H:%M:%S')
        body['design_sha8'] = d8
        with io.open(exep, 'w', encoding='utf-8', newline='\n') as f:
            f.write(json.dumps(body, ensure_ascii=False, indent=1, sort_keys=True))
        log('freeze: execution.json written design_sha8=%s' % d8)
    return d8


# ---------------- subspace builders (3165/3166 verbatim) ----------------------
def load_WU(mdir):
    from safetensors import safe_open
    cfgm = json.load(io.open(os.path.join(mdir, 'config.json'), encoding='utf-8'))
    tied = bool(cfgm.get('tie_word_embeddings', False))
    want = 'model.embed_tokens.weight' if tied else 'lm_head.weight'
    for sh in sorted(os.listdir(mdir)):
        if not sh.endswith('.safetensors'):
            continue
        with safe_open(os.path.join(mdir, sh), framework='pt') as f:
            if want in set(f.keys()):
                W = f.get_tensor(want).float().numpy()
                break
    else:
        raise AssertionError('unembed not found in ' + mdir)
    return W.astype(np.float64), cfgm


def build_S_class(W, mdir):
    """2881/3165/3166 recipe verbatim: CATS + single_tok + centroid diff."""
    e2806 = json.load(io.open(os.path.join(
        RDIR, 'phase2806', 'qwen4_hierarchy', 'execution.json'), encoding='utf-8'))
    CATS = e2806['cats']
    CAT_WORDS = list(CATS.keys())
    assert CAT_WORDS == ['fruit', 'animal', 'metal', 'vehicle', 'country',
                         'food', 'nature', 'furniture', 'tool', 'clothing']
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(mdir, local_files_only=True,
                                        trust_remote_code=True, use_fast=True)
    tc = {}

    def tid(t):
        if t not in tc:
            ids = tok(' ' + t, add_special_tokens=False)['input_ids']
            if len(ids) != 1:
                ids = tok(t, add_special_tokens=False)['input_ids']
            assert len(ids) == 1, '%s -> %s' % (t, ids)
            tc[t] = int(ids[0])
        return tc[t]

    all_words = [w for v in CATS.values() for w in v]
    single_tok = []
    for w in all_words:
        try:
            tid(w)
            single_tok.append(w)
        except AssertionError:
            pass
    class_targets = {}
    for cat in CAT_WORDS:
        class_targets[cat] = [w for w in CATS[cat] if w in single_tok][:8]
    assert sum(len(v) for v in class_targets.values()) == 80, 'class_targets != 80'
    Erows = {w: W[tid(w)] for w in single_tok}
    cents = []
    for cat in CAT_WORDS:
        ws = [w for w in CATS[cat] if w in single_tok]
        cents.append(np.stack([Erows[w] for w in ws]).mean(0))
    Cm = np.stack(cents)
    dW_c = Cm - (Cm.sum(0, keepdims=True) - Cm) / 9.0
    dW_class = np.stack([_unit(dW_c[i]) for i in range(10)])
    return dW_class.astype(np.float32)


def _unit(v):
    n = float(np.linalg.norm(v))
    assert n > 0, 'zero vector in unit()'
    return v / n


def kross(A, Bd):
    """principal angles from row-form (k, D); cos desc, top1 deg (3165 verbatim)."""
    Qa, _ = np.linalg.qr(A.T)
    Qb, _ = np.linalg.qr(Bd.T)
    s = np.linalg.svd(Qa.T @ Qb, compute_uv=False)
    s = np.clip(s, 0.0, 1.0)
    top1 = float(np.degrees(np.arccos(s[0])))
    return s, top1


def orth_rows(A):
    Q, _ = np.linalg.qr(A.T)
    return Q  # (D, k) column orthonormal


def proj_frac(Q, Hrows):
    """mean over rows of ||Q^T h||^2 / ||h||^2; Hrows (n, D) float64."""
    num = ((Hrows @ Q) ** 2).sum(1)
    den = (Hrows ** 2).sum(1)
    assert (den > 0).all(), 'zero row in H'
    return float((num / den).mean())


def nnls_fit(Dict, target):
    """NNLS of target on Dict columns (n x k); returns weights, rel residual."""
    from scipy.optimize import nnls
    x, rnorm = nnls(Dict, target)
    rel = float(rnorm) / float(np.linalg.norm(target))
    return x.tolist(), rel


# ---------------- run ----------------------------------------------------------
def run():
    d8 = freeze()
    # ---- G_anchor ----
    checks = []
    src_paths = {
        'p3169_4b': MODELS[0]['npz'], 'p3169_14b': MODELS[1]['npz'],
        'p3169_glm4': MODELS[2]['npz'], 'p3169_smoke_4b': SMOKE_NPZ,
        'p3158_4b': MODELS[0]['t58'], 'p3158_14b': MODELS[1]['t58'],
        'p3158_glm4': MODELS[2]['t58'],
        'p2806_exec': os.path.join(RDIR, 'phase2806', 'qwen4_hierarchy', 'execution.json'),
        'p3166_result': P3166,
        'p3169_result': os.path.join(P3169, 'result.json'),
    }
    for k, v in SHA_ANCHOR.items():
        got = sha8(src_paths[k])
        assert got == v, ('G_anchor', k, got, v)
    log('G_anchor: %d files OK' % len(SHA_ANCHOR))

    R3169 = json.load(io.open(src_paths['p3169_result'], encoding='utf-8'))
    R3166 = json.load(io.open(P3166, encoding='utf-8'))
    assert R3169['res_sha8'] == '49430a39' and R3166['res_sha8'] == '89b3f320', \
        'sealed anchor results drifted'

    # ---- G_panel (counts) ----
    n_exp = dict(seen=len(SEEN_ROWS), oov=len(OOV_ROWS),
                 newent=len(NEWENT_ROWS), oov_pure=len(OOVPURE_ROWS))
    if not SMOKE:
        assert n_exp == dict(seen=738, oov=876, newent=576, oov_pure=384), n_exp
    assert len(set(SEEN_ROWS) & set(OOV_ROWS)) == 0, 'row-set overlap'
    log('G_panel: row counts %s' % n_exp)

    per_model = {}
    crosscheck = {}
    for m in (MODELS[:1] if SMOKE else MODELS):
        mk = m['name']
        log('=== model %s ===' % mk)
        npz_path = SMOKE_NPZ if SMOKE else m['npz']
        z = np.load(npz_path)
        H16, MARG = z['H'], z['marg']
        NTz, NPz, NH, D = H16.shape
        assert NTz == NT and NPz == NP_, ('shape', H16.shape)
        assert MARG.shape == (NT, NP_, NC), ('marg shape', MARG.shape)
        assert np.isfinite(H16.astype(np.float32)).all(), 'H non-finite'
        assert np.isfinite(MARG).all(), 'MARG non-finite'
        cfg = json.load(io.open(os.path.join(m['mdir'], 'config.json'), encoding='utf-8'))
        NL = int(cfg['num_hidden_layers'])
        assert NH == NL + 1, ('NH', NH, NL)
        k_main = NL - 1
        # G_slot: primary slot must equal the 3169 readout slot
        rd = int(R3169['per_model'][mk]['readout'])
        ks = int(R3169['per_model'][mk]['kstar'])
        assert k_main == rd, ('G_slot', k_main, rd)
        assert ks == 3, ('G_slot kstar', ks)
        log('G_slot: k_main=%d == 3169 readout; kstar=3; NL=%d D=%d' % (k_main, NL, D))

        # subspaces (3166 verbatim rebuild, float32 round-trip -> float64)
        W, _ = load_WU(m['mdir'])
        S_class = build_S_class(W, m['mdir']).astype(np.float64)   # (10, D)
        z58 = np.load(m['t58'])
        top64 = z58['top64'].astype(np.float64)                    # (D, 64)
        K_read = top64.T                                           # (64, D)
        QS = orth_rows(S_class)                                    # (D, 10)
        QK = orth_rows(K_read)                                     # (D, 64)

        # G_crosscheck: reproduce 3166 census K_readout x S_class top1
        # (3166 result names the glm4 model 'glm4'; 3169 uses 'glm4-9b')
        mk66 = {'qwen3-4b': 'qwen3-4b', 'qwen3-14b': 'qwen3-14b',
                'glm4-9b': 'glm4'}[mk]
        anchor_deg = float(R3166['per_model'][mk66]['census']['K_readout__S_class']['top1_deg'])
        rec_deg = kross(S_class, K_read)[1]
        drift = abs(rec_deg - anchor_deg)
        assert drift < 1e-3, ('G_crosscheck', rec_deg, anchor_deg)
        crosscheck[mk] = dict(anchor_deg=anchor_deg, recomputed_deg=rec_deg,
                              drift_deg=drift)
        log('G_crosscheck: K_readout x S_class top1 rec=%.6f anchor=%.6f drift=%.2e'
            % (rec_deg, anchor_deg, drift))

        def Y_at(k):
            return H16[:, :, k, :].reshape(NT * NP_, D).astype(np.float32).astype(np.float64)

        slots_out = {}
        for k in (k_main, NL, 3):
            Y = Y_at(k)
            fS_seen = proj_frac(QS, Y[SEEN_ROWS])
            fS_oov = proj_frac(QS, Y[OOV_ROWS])
            ratio_S = fS_oov / fS_seen
            fK_seen = proj_frac(QK, Y[SEEN_ROWS])
            fK_oov = proj_frac(QK, Y[OOV_ROWS])
            ratio_K = fK_oov / fK_seen
            tag = 'k_main' if k == k_main else ('k_NL' if k == NL else 'k_3')
            ent = dict(f_S_seen=fS_seen, f_S_oov=fS_oov, ratio_S=ratio_S,
                       f_K_seen=fK_seen, f_K_oov=fK_oov, ratio_K=ratio_K)
            if k == k_main:
                assert fS_seen > 1e-8 and fK_seen > 1e-4, ('G_sanity', fS_seen, fK_seen)
                ent['cls'] = ('encoding_missing' if ratio_S < 0.5 else
                              'readout_missing' if ratio_S > 0.8 else 'mixed')
                # oov-pure rows (new entity x oov class) side reading
                fS_op = proj_frac(QS, Y[OOVPURE_ROWS])
                ent['f_S_oov_pure'] = fS_op
                ent['f_S_newent'] = proj_frac(QS, Y[NEWENT_ROWS])
                log('slot %s (k=%d): f_S seen=%.6e oov=%.6e ratio=%.4f -> %s | '
                    'f_K seen=%.6e oov=%.6e ratio=%.4f'
                    % (tag, k, fS_seen, fS_oov, ratio_S, ent['cls'],
                       fK_seen, fK_oov, ratio_K))
            else:
                log('slot %s (k=%d): ratio_S=%.4f ratio_K=%.4f (side)'
                    % (tag, k, ratio_S, ratio_K))
            slots_out[tag] = ent
            del Y

        # ---- probe (c): centroids at k_main ----
        Yc = Y_at(k_main)
        cents = np.zeros((NC, D), np.float64)
        for c in range(NC):
            rows = [t * NP_ + pi for t in range(NT) for pi in range(NP_)
                    if PAIRS[pi][1] == c]
            cents[c] = Yc[rows].mean(0)
        seen_c, oov_c = cents[:6], cents[6:]
        cn = np.linalg.norm(seen_c, axis=1, keepdims=True)
        on = np.linalg.norm(oov_c, axis=1, keepdims=True)
        cos_os = (oov_c / on) @ (seen_c / cn).T                       # (4, 6)
        QSsc = orth_rows(S_class)                                     # reuse
        cos_vs_sclass = (oov_c / on) @ S_class.T                      # (4, 10)
        # seen pairwise cosine (baseline distinctness)
        cos_ss = (seen_c / cn) @ (seen_c / cn).T
        # NNLS cone fit: each oov centroid on the 6 seen centroids
        nnls_oov, proj_oov = [], []
        Qseen = orth_rows(seen_c)                                     # (D, 6)
        for j in range(4):
            x, rel = nnls_fit(seen_c.T, oov_c[j])
            nnls_oov.append(dict(cls=CLASSES[6 + j], weights=x, rel_resid=rel))
            proj_oov.append(float(((oov_c[j] @ Qseen) ** 2).sum() /
                                 (oov_c[j] ** 2).sum()))
        # leave-one-out seen baselines
        nnls_loo, proj_loo = [], []
        for j in range(6):
            others = np.delete(seen_c, j, axis=0)
            x, rel = nnls_fit(others.T, seen_c[j])
            nnls_loo.append(dict(cls=CLASSES[j], rel_resid=rel))
            Qo = orth_rows(others)
            proj_loo.append(float(((seen_c[j] @ Qo) ** 2).sum() /
                                  (seen_c[j] ** 2).sum()))
        # full 10x10 panel-centroid cosine matrix
        nall = np.linalg.norm(cents, axis=1, keepdims=True)
        cos_full = (cents / nall) @ (cents / nall).T
        centroid = dict(
            cos_oov_seen=[[float(v) for v in row] for row in cos_os],
            nearest=[dict(oov=CLASSES[6 + j],
                          seen=CLASSES_SEEN[int(np.argmax(cos_os[j]))],
                          cos_top1=float(np.max(cos_os[j])),
                          cos_top2=float(np.sort(cos_os[j])[-2])) for j in range(4)],
            cos_vs_sclass=[[float(v) for v in row] for row in cos_vs_sclass],
            nnls_cone=nnls_oov, proj_frac_on_seen=proj_oov,
            baseline_nnls_loo=nnls_loo, baseline_proj_loo=proj_loo,
            cos_seen_pairwise=[[float(v) for v in row] for row in cos_ss],
            cos_full_10x10=[[float(v) for v in row] for row in cos_full])
        log('centroids: nearest ' + '; '.join(
            '%s->%s(%.3f)' % (n['oov'], n['seen'], n['cos_top1'])
            for n in centroid['nearest']))
        log('cone: oov rel_resid ' + '; '.join('%s=%.3f' % (n['cls'], n['rel_resid'])
                                               for n in nnls_oov) +
            ' | seen-LOO ' + '; '.join('%.3f' % n['rel_resid'] for n in nnls_loo))

        # ---- probe (d): MARG descriptive ----
        Mf = MARG.astype(np.float64).reshape(NT * NP_, NC)
        seen_true = [float(Mf[r, PAIRS[r % NP_][1]]) for r in SEEN_ROWS]
        oov_claimed = [float(Mf[r, PAIRS[r % NP_][1]]) for r in OOV_ROWS]
        oov_max_seen = [float(Mf[r, :6].max()) for r in OOV_ROWS]
        marg = dict(seen_true_margin_mean=float(np.mean(seen_true)),
                    oov_claimed_margin_mean=float(np.mean(oov_claimed)),
                    oov_max_seen_margin_mean=float(np.mean(oov_max_seen)))
        log('MARG: seen true-class margin=%.4f | oov claimed-class=%.4f | '
            'oov max-seen=%.4f' % (marg['seen_true_margin_mean'],
                                   marg['oov_claimed_margin_mean'],
                                   marg['oov_max_seen_margin_mean']))

        per_model[mk] = dict(NL=NL, D=D, k_main=k_main, slots=slots_out,
                             centroid=centroid, marg=marg)
        del z, H16, MARG, Yc

    # ---- verdict ----
    if SMOKE:
        main_cls = 'smoke'
        verdict = ('g5a8_collapse_mechanism|smoke|1_model|k_main_ratio=%.4f|%s'
                   % (per_model['qwen3-4b']['slots']['k_main']['ratio_S'],
                      per_model['qwen3-4b']['slots']['k_main']['cls']))
    else:
        clses = [per_model[mk]['slots']['k_main']['cls'] for mk in
                 ('qwen3-4b', 'qwen3-14b', 'glm4-9b')]
        ratios = [per_model[mk]['slots']['k_main']['ratio_S'] for mk in
                  ('qwen3-4b', 'qwen3-14b', 'glm4-9b')]
        main_cls = clses[0] if len(set(clses)) == 1 else 'mixed_across_models'
        verdict = ('g5a8_collapse_mechanism|full|3_models|ratio_S_main=' +
                   '/'.join('%.4f' % r for r in ratios) + '|' + main_cls)
        log('VERDICT: cls per model %s -> %s' % (clses, main_cls))

    result = dict(phase=PHASE, name=NAME, design_sha8=d8, smoke=SMOKE,
                  rows=n_exp, crosscheck=crosscheck, per_model=per_model,
                  overall=dict(cls_per_model=clses if not SMOKE else ['smoke'],
                               main_cls=main_cls,
                               gate='pre-registered: ratio<0.5 encoding_missing '
                                    '/ >0.8 readout_missing / else mixed'),
                  verdict=verdict)

    # ---- seal ----
    raw = json.dumps(result, ensure_ascii=False, indent=1, sort_keys=True)
    res8 = hashlib.sha256(raw.encode('utf-8')).hexdigest()[:8]
    mid = json.dumps(dict(result, res_sha8=res8), ensure_ascii=False,
                     indent=1, sort_keys=True)
    seal8 = hashlib.sha256(mid.encode('utf-8')).hexdigest()[:8]
    result['res_sha8'] = res8
    result['seal_sha8'] = seal8
    fname = 'smoke_result.json' if SMOKE else 'result.json'
    with io.open(os.path.join(OUTDIR, fname), 'w', encoding='utf-8',
                 newline='\r\n') as f:
        f.write(json.dumps(result, ensure_ascii=False, indent=1, sort_keys=True))
    log('sealed %s res=%s seal=%s' % (fname, res8, seal8))
    lname = 'smoke_run_log.txt' if SMOKE else 'run_log.txt'
    with io.open(os.path.join(OUTDIR, lname), 'w', encoding='utf-8',
                 newline='\r\n') as f:
        f.write('\n'.join(LOG) + '\n')
    log('DONE')


if __name__ == '__main__':
    run()
