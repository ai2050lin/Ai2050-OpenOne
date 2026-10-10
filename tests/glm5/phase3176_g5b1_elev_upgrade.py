# -*- coding: utf-8 -*-
"""Phase 3176: G5-B1 E1->E2 zero-GPU batch upgrade + atlas v1.4 render.

Per prereg in MEMO Phase 3175 tail (and 3174 ranking item 2):
  (a) FTR-04 held_out arm: S_class (3166 recipe, per-model W_U rebuild) LOEO
      class readout on the 3169 panel seen rows (35 usable entities with true
      class in the 5 mappable p2806 categories; chance = 1/5 = 0.2; random
      10-dim orthonormal subspace as descriptive control);
      gate: LOEO acc >= 0.5 for all 3 models -> held_out anchor appended,
      FTR-04 upgraded E1_repeatable -> E2_predictive;
  (b) FTR-08 held_out arm: R_logic LOEO (leave-one-entity-out rebuild of the
      per-entity contrastive top8 over the sealed 3151/3152 true/false panel,
      41 folds) x fixed K_entity (3157 verbatim); gate: |loeo_mean - 3166
      census anchor| < 5 deg AND classification band agrees per model ->
      held_out anchor appended, FTR-08 upgraded to E2_predictive;
  (c) FTR-06 cross_model arm: unembed-only per-model rebuild of S_attr (2874
      pairs_json axes recipe verbatim: kept single-token words, unit pair
      diffs, axis dir = unit(mean)) and S_syntax (2878); matrix = per-model
      W_U (3166 load chain; 4b tie => same matrix as 2874 embed_tokens);
      K_readout = p3158 top64 per model; 4b rebuild cross-checked vs 3165
      anchors 68.094/58.488 (tol 0.5 deg);
      gate: 14b and glm4 top1 >= 30 deg (separable band reproduced) ->
      cross_model anchor appended, FTR-06 upgraded to E2_predictive,
      model_scope 4b -> 3 models;
  (d) registry v1.2 -> v1.3 (three upgraded features replaced, other 19 +
      failures 14 byte-for-byte, upgrade_log += 3 events) + gap ledger v1.5
      (3175 sealed, rendered as-is) + atlas_v1_4.html full re-render with
      per-field data-k verification.

Failure branch (prereg): any arm gate failing -> that feature stays E1, an
F15 entry is appended to failures, no upgrade_log event for it; render and
versioning proceed regardless (honest registry).

SMOKE (env P3176_SMOKE=1): the three arms always run FULL (zero-GPU, few
minutes; no SMOKE truncation => numbers identical between modes); only the
html render subset differs.  DESIGN is fully static (3169 discipline).
"""
import io
import os
import re
import json
import copy
import html
import hashlib
import datetime

import numpy as np

BASE = r'D:\AI2050\Ai2050-OpenOne'
SRC_DIR = os.path.join(BASE, r'tests\glm5\result\rdc_query_construction_20260913')
OUTDIR = os.path.join(SRC_DIR, 'phase3176', 'g5b1_elev_upgrade')
SMOKE = os.environ.get('P3176_SMOKE', '') == '1'
MODELS_HF = {'qwen3-4b': 'qwen3-4b', 'qwen3-14b': 'Qwen3-14B', 'glm4-9b': 'glm4-9b-chat-hf'}

SRC = {
    'p3151_collect': os.path.join(SRC_DIR, r'phase3151\g1p1_combo_additive_vs_interaction\collect.npz'),
    'p3152_4b': os.path.join(SRC_DIR, r'phase3152\g1p2_tri_model_k1\qwen3-4b\collect.npz'),
    'p3152_14b': os.path.join(SRC_DIR, r'phase3152\g1p2_tri_model_k1\qwen3-14b\collect.npz'),
    'p3157_4b': os.path.join(SRC_DIR, r'phase3157\g2p2_transform_algebra_commutator\qwen3-4b\collect.npz'),
    'p3157_14b': os.path.join(SRC_DIR, r'phase3157\g2p2_transform_algebra_commutator\qwen3-14b\collect.npz'),
    'p3157_glm4': os.path.join(SRC_DIR, r'phase3157\g2p2_transform_algebra_commutator\glm4\collect.npz'),
    'p3158_4b': os.path.join(SRC_DIR, r'phase3158\g4p1_output_equivalence_class\qwen3-4b\collect.npz'),
    'p3158_14b': os.path.join(SRC_DIR, r'phase3158\g4p1_output_equivalence_class\qwen3-14b\collect.npz'),
    'p3158_glm4': os.path.join(SRC_DIR, r'phase3158\g4p1_output_equivalence_class\glm4\collect.npz'),
    'p3165_result': os.path.join(SRC_DIR, r'phase3165\g5a3_family_alignment\result.json'),
    'p3166_result': os.path.join(SRC_DIR, r'phase3166\g5a3b_logic_direction\result.json'),
    'p3169_4b': os.path.join(SRC_DIR, r'phase3169\g5a6_oov_panel\collect_qwen3-4b.npz'),
    'p3169_14b': os.path.join(SRC_DIR, r'phase3169\g5a6_oov_panel\collect_qwen3-14b.npz'),
    'p3169_glm4': os.path.join(SRC_DIR, r'phase3169\g5a6_oov_panel\collect_glm4-9b.npz'),
    'p3169_result': os.path.join(SRC_DIR, r'phase3169\g5a6_oov_panel\result.json'),
    'p2874': os.path.join(SRC_DIR, r'phase2874\attr_vocab_v2\attr_vocab_v2.npz'),
    'p2878': os.path.join(SRC_DIR, r'phase2878\syntax_trans_vocab\syntax_trans_vocab.npz'),
    'p2806_exec': os.path.join(SRC_DIR, r'phase2806\qwen4_hierarchy\execution.json'),
    'registry_v12': os.path.join(SRC_DIR, r'phase3173\g5a10_atlas_v13\atlas_registry_v1_2.json'),
    'gap_v15': os.path.join(SRC_DIR, r'phase3175\g5a12_residual_arms\gap_ledger_v1_5.json'),
}
SHA_ANCHOR = {
    'p3151_collect': 'c711946c', 'p3152_4b': '4ef190b7', 'p3152_14b': '733182c1',
    'p3157_4b': '95a25965', 'p3157_14b': '9552086d', 'p3157_glm4': '23dd74eb',
    'p3158_4b': 'c4d9b9fb', 'p3158_14b': '7f53f3a0', 'p3158_glm4': '2ea72cfe',
    'p3165_result': '511d9b13', 'p3166_result': '448af595',
    'p3169_4b': '36eb4ff0', 'p3169_14b': '97f98575', 'p3169_glm4': '9cc3f8b4',
    'p3169_result': '5b51c2c1',
    'p2874': '5f7796e1', 'p2878': '0bb3ec1e', 'p2806_exec': 'b291ea35',
    'registry_v12': '0e5abcaf', 'gap_v15': '4fb3f37d',
}
CONTENT_SHA = {'p3165': ('9b0fe9c9', 'e966a4e5'), 'p3166': ('89b3f320', '263de0ab'),
               'p3169': ('49430a39', '018c6024')}

# ---------------- 3151/3152/3166 frozen panel verbatim ----------------
CLASSES = ['水果', '动物', '交通工具', '家具', '金属', '颜色']
ENT = {
    '水果': ['苹果', '香蕉', '梨', '西瓜', '葡萄', '草莓', '芒果', '柠檬'],
    '动物': ['狗', '猫', '老虎', '大象', '兔子', '猴子', '马', '牛'],
    '交通工具': ['汽车', '火车', '飞机', '摩托车', '卡车', '地铁'],
    '家具': ['桌子', '椅子', '床', '沙发', '地毯', '窗帘'],
    '金属': ['铁', '铜', '铝', '金', '银', '锌', '铅'],
    '颜色': ['红', '蓝', '绿', '黄', '黑', '白'],
}
ENTS = [e for cl in CLASSES for e in ENT[cl]]
CLS_OF = [CLASSES.index(cl) for cl in CLASSES for e in ENT[cl]]
NE = len(ENTS)                      # 41
NC = len(CLASSES)                   # 6
PAIRS_FULL = [(i, c) for i in range(NE) for c in range(NC)]   # 246

# 3169 panel (3172 verbatim)
CLASSES_3169 = CLASSES + ['乐器', '天气', '运动', '电器']
NE69, NC69 = 73, 10
NP69 = NE69 * NC69                  # 730 pairs; H row = t*730 + pi; pi = i*10 + c
SEEN_CLS_MAP = {'水果': 'fruit', '动物': 'animal', '交通工具': 'vehicle',
                '家具': 'furniture', '金属': 'metal'}
CLS69_IDX = {cn: i for i, cn in enumerate(CLASSES_3169)}

MODELS = ['qwen3-4b', 'qwen3-14b', 'glm4-9b']
MODEL_NPZ = {'qwen3-4b': ('p3152_4b', 'p3157_4b', 'p3158_4b', 'p3169_4b'),
             'qwen3-14b': ('p3152_14b', 'p3157_14b', 'p3158_14b', 'p3169_14b'),
             'glm4-9b': ('p3151_collect', 'p3157_glm4', 'p3158_glm4', 'p3169_glm4')}
R3166_KEY = {'qwen3-4b': 'qwen3-4b', 'qwen3-14b': 'qwen3-14b', 'glm4-9b': 'glm4'}
CENSUS_ANCHOR_RK = {'qwen3-4b': 45.845, 'qwen3-14b': 26.641, 'glm4-9b': 46.005}
ARM_A_GATE = 0.5
ARM_B_TOL_DEG = 5.0
ARM_C_TOL_4B = 0.5

DESIGN = {
    'phase': 3176, 'name': 'g5b1_elev_upgrade', 'zero_gpu': True,
    'sources': {k: v.replace(BASE, '') for k, v in SRC.items()},
    'arm_a': dict(
        feature='FTR-04', upgrade='E1_repeatable -> E2_predictive (held_out)',
        protocol='S_class rebuilt per model (3166 verbatim: p2806 CATS, per-model '
                 'W_U, single-token words, centroid diff, float32->float64); 3169 '
                 'panel seen rows H[t*730+pi, NL, :], pi = i*10 + c, i < 41, c < 6; '
                 'usable entities = 35 (true class in the 5 mappable p2806 cats: '
                 'fruit/animal/vehicle/furniture/metal; color has no p2806 axis '
                 'and is excluded from folds and centroids); LOEO fold e: train = '
                 'true-class rows of the other 34 usable entities (3 templates '
                 'each), class centroids in the 10-dim S_class projection; test = '
                 'the 15 rows of e over the 5 mappable classes x 3 templates; '
                 'prediction = argmax cosine to centroid; acc = mean over all '
                 '525 test rows; control = random 10-dim orthonormal subspace '
                 '(RandomState(0)) same protocol, descriptive only',
        gate='LOEO acc >= 0.5 for all 3 models (chance 0.2)'),
    'arm_b': dict(
        feature='FTR-08', upgrade='E1_repeatable -> E2_predictive (held_out)',
        protocol='panel = sealed 3151/3152 true/false H (3, 246, NL+1, D), rows '
                 't*246+pi, y = (c == CLS_OF[i]); R_logic rebuild verbatim 3166 '
                 'build_R (per-entity true-false mean diff, unit, center, SVD '
                 'top8); K_entity fixed from 3157 (H[:, NL57, :] center SVD top8); '
                 'LOEO: fold e rebuilds top8 from the other 40 entities, angle = '
                 'top1 principal angle vs fixed K_entity; 41 folds per model; '
                 'full-sample rebuild cross-checked vs 3166 census anchor '
                 '(round 3, tol 0.05 deg)',
        gate='|mean(LOEO angles) - 3166 census anchor| < 5 deg AND '
             'cls_of_top1 band agrees, for all 3 models'),
    'arm_c': dict(
        feature='FTR-06', upgrade='E1_repeatable -> E2_predictive (cross_model, '
                                  'model_scope 4b -> 3 models)',
        protocol='AXES from p2874/p2878 pairs_json (verbatim vocab source); per '
                 'model: kept = single-token words under that tokenizer; valid '
                 'pair needs both poles kept; axis dir = unit(mean(unit(E[w+] - '
                 'E[w-]))); axes with < 2 valid pairs dropped and logged; matrix '
                 '= per-model W_U (3166 load chain: tied ? embed_tokens : '
                 'lm_head; 4b tie => identical matrix to the 2874 embed_tokens '
                 'build); K_readout = p3158 top64 per model; top1 principal '
                 'angles K_readout x S_attr / x S_syntax',
        gate='4b rebuild within 0.5 deg of 3165 anchors (68.094 / 58.488); '
             '14b and glm4 top1 >= 30 deg on both S_attr and S_syntax'),
    'updates': {
        'registry': 'v1.2 -> v1.3: FTR-04/FTR-06/FTR-08 upgraded per arm verdicts '
                    '(E2_predictive, new anchors with held_out/cross_model tags, '
                    'values extended, statement extended); other 19 features + '
                    'failures 14 byte-for-byte; upgrade_log += up to 3 events '
                    '(one per successful arm); failing arm -> feature stays E1 + '
                    'F15 failure entry.',
        'gap_ledger': 'no new version: gap ledger v1.5 (3175 sealed, 4fb3f37d) is '
                      'rendered as-is (GAP-4 finalized in 3175).',
        'html': 'atlas_v1_4.html full re-render (22 features, 16 nodes, GAP-4 incl '
                'mechanism_note, failures F1-F14) with per-field data-k '
                'verification against flattened v1.3/v1.5 sources.',
    },
    'gates': {
        'G1': 'html data-k field check: missing/extra/mismatch/dup all 0',
        'G2': 'feature cards rendered (22 full / 3 smoke)',
        'G3': 'v1.2 other 19 features + failures 14 byte-for-byte; upgrade_log '
              'first 3 byte-for-byte; gap v1.5 file sha anchored',
        'G4': 'upgrade asserts: evidence_level E2_predictive on successful arms, '
              'new anchor tags held_out/cross_model, values == runtime numbers',
        'G5': 'arm gates (a/b/c) evaluated and logged per model',
        'G6': 'no external resources in html',
        'G7': 'upgrade_log total = 3 + n_upgraded',
    },
    'smoke': {'features': 3, 'nodes': 4, 'failures': 2, 'upgrades': 'all',
              'arms': 'full (no truncation; numbers identical across modes)'},
}

LOG = []


def log(s):
    LOG.append(str(s))
    print(s, flush=True)


def sha8(b):
    return hashlib.sha256(b).hexdigest()[:8]


def sha8_file(p):
    return sha8(open(p, 'rb').read())


def fmt(v):
    if isinstance(v, bool):
        return 'true' if v else 'false'
    if isinstance(v, float):
        return '%.6g' % v
    return str(v)


def assert_val(tag, actual, expect, tol=1e-6):
    if isinstance(expect, float) and isinstance(actual, (int, float)) and not isinstance(actual, bool):
        ok = abs(float(actual) - expect) <= tol * max(1.0, abs(expect))
    elif isinstance(expect, list):
        ok = list(actual) == list(expect)
    else:
        ok = actual == expect
    assert ok, ('assert fail', tag, actual, expect)
    return ok


def freeze():
    os.makedirs(OUTDIR, exist_ok=True)
    exep = os.path.join(OUTDIR, 'execution.json')
    body = dict(DESIGN)
    raw = json.dumps(body, ensure_ascii=False, indent=1, sort_keys=True)
    d8 = sha8(raw.encode('utf-8'))
    body['design_sha8'] = d8
    raw2 = json.dumps(body, ensure_ascii=False, indent=1, sort_keys=True)
    if os.path.exists(exep):
        prev = json.load(io.open(exep, encoding='utf-8'))
        if prev.get('design_sha8') != d8:
            raise SystemExit('DRIFT: execution.json design_sha8 %s != current %s'
                             % (prev.get('design_sha8'), d8))
        log('freeze: existing execution.json OK (%s)' % d8)
    else:
        with io.open(exep, 'w', encoding='utf-8') as f:
            f.write(raw2)
        log('freeze: execution.json written design_sha8=%s' % d8)
    return d8


# ------------------------------------------------------- shared recipes (verbatim)
def unit(v):
    return v / max(float(np.linalg.norm(v)), 1e-30)


def load_WU(mdir):
    cfgm = json.load(io.open(os.path.join(mdir, 'config.json'), encoding='utf-8'))
    tied = bool(cfgm.get('tie_word_embeddings', False))
    want = 'model.embed_tokens.weight' if tied else 'lm_head.weight'
    idx_path = os.path.join(mdir, 'model.safetensors.index.json')
    if os.path.exists(idx_path):
        idx = json.load(io.open(idx_path, encoding='utf-8'))
        shard = idx['weight_map'][want]
    else:
        shard = 'model.safetensors'
    from safetensors import safe_open
    with safe_open(os.path.join(mdir, shard), framework='pt') as f:
        W = f.get_tensor(want).float().numpy()
    return W, cfgm


class TidCache(object):
    def __init__(self, tok):
        self.tok = tok
        self.tc = {}

    def tid(self, t):
        if t not in self.tc:
            ids = self.tok(' ' + t, add_special_tokens=False)['input_ids']
            if len(ids) != 1:
                ids = self.tok(t, add_special_tokens=False)['input_ids']
            assert len(ids) == 1, '%s -> %s' % (t, ids)
            self.tc[t] = int(ids[0])
        return self.tc[t]


def kross(A, Bd):
    Qa, _ = np.linalg.qr(A.T)
    Qb, _ = np.linalg.qr(Bd.T)
    s = np.linalg.svd(Qa.T @ Qb, compute_uv=False)
    s = np.clip(s, 0.0, 1.0)
    top1 = float(np.degrees(np.arccos(s[0])))
    return s, top1


def effdim(s):
    c2 = s ** 2
    cnt = int((c2 >= 0.5).sum())
    pr = float(c2.sum() ** 2 / (c2 ** 2).sum() + 1e-300)
    return cnt, pr


def cls_of_top1(t1):
    return 'separable' if t1 >= 30.0 else ('collinear' if t1 < 15.0 else 'weakly_separated')


def build_S_class(W, tidc):
    """3166 verbatim: p2806 CATS + single_tok + centroid diff (float32 out)."""
    e2806 = json.load(io.open(SRC['p2806_exec'], encoding='utf-8'))
    CATS = e2806['cats']
    CAT_WORDS = list(CATS.keys())
    assert CAT_WORDS == ['fruit', 'animal', 'metal', 'vehicle', 'country',
                         'food', 'nature', 'furniture', 'tool', 'clothing']
    all_words = [w for v in CATS.values() for w in v]
    single_tok = []
    for w in all_words:
        try:
            tidc.tid(w)
            single_tok.append(w)
        except AssertionError:
            pass
    Erows = {w: W[tidc.tid(w)] for w in single_tok}
    cents = []
    for cat in CAT_WORDS:
        ws = [w for w in CATS[cat] if w in single_tok]
        cents.append(np.stack([Erows[w] for w in ws]).mean(0))
    Cm = np.stack(cents)
    dW_c = Cm - (Cm.sum(0, keepdims=True) - Cm) / 9.0
    dW_class = np.stack([unit(dW_c[i]) for i in range(10)])
    return dW_class.astype(np.float32), single_tok


def build_R(Hk, ent_rows, y, ents):
    """3166 verbatim: per-entity contrastive -> unit -> center -> SVD top8."""
    ds = []
    for i in ents:
        tr = [r for r in ent_rows[i] if y[r]]
        fa = [r for r in ent_rows[i] if not y[r]]
        assert tr and fa, ('empty arm', i)
        ds.append(Hk[tr].mean(0) - Hk[fa].mean(0))
    Dm = np.stack(ds)
    DmU = np.stack([unit(v) for v in Dm])
    Dc = DmU - DmU.mean(0, keepdims=True)
    _, sv, Vh = np.linalg.svd(Dc, full_matrices=False)
    return Vh[:8]


# ------------------------------------------------------- arm (a)
def arm_a(tidc_by_model, W_rows_by_model):
    """LOEO class readout via S_class on the 3169 panel seen rows."""
    r69 = json.load(io.open(SRC['p3169_result'], encoding='utf-8'))
    assert r69['res_sha8'] == CONTENT_SHA['p3169'][0]
    assert r69['seal_sha8'] == CONTENT_SHA['p3169'][1]
    assert_val('r69 gate.ratio_B', r69['gate']['ratio_B'], 2.5388114997805062, tol=1e-12)
    usable = [i for i in range(NE) if CLASSES[CLS_OF[i]] in SEEN_CLS_MAP]
    assert len(usable) == 35, ('usable ents', len(usable))
    cat_c69 = {}
    for cn69, cn in SEEN_CLS_MAP.items():
        cat_c69[cn] = CLS69_IDX[cn69]
    out = {}
    for m in MODELS:
        z = np.load(SRC[MODEL_NPZ[m][3]])
        H = z['H']
        NTn, NPn, NHn, Dn = H.shape
        assert (NTn, NPn) == (3, NP69), ('3169 panel shape', H.shape)
        NL = NHn - 1
        Hk = H[:, :, NL, :].reshape(NTn * NPn, Dn).astype(np.float64)
        S_class = build_S_class(W_rows_by_model[m]['W'], tidc_by_model[m])[0].astype(np.float64)
        rng = np.random.RandomState(0)
        Qrand, _ = np.linalg.qr(rng.randn(Dn, 10))
        accs, accs_rand = [], []
        for e in usable:
            true_cn = SEEN_CLS_MAP[CLASSES[CLS_OF[e]]]
            train_rows, labels = [], []
            for i in usable:
                if i == e:
                    continue
                r0 = i * NC69 + CLS69_IDX[CLASSES[CLS_OF[i]]]
                cn = SEEN_CLS_MAP[CLASSES[CLS_OF[i]]]
                for t in range(3):
                    train_rows.append(t * NP69 + r0)
                    labels.append(cn)
            test_rows = [t * NP69 + e * NC69 + c
                         for t in range(3) for c in sorted(cat_c69.values())]
            P = Hk[train_rows] @ S_class.T
            Pr = Hk[train_rows] @ Qrand
            mus = {cn: np.stack([P[j] for j in range(len(train_rows)) if labels[j] == cn]).mean(0)
                   for cn in sorted(set(labels))}
            mus_r = {cn: np.stack([Pr[j] for j in range(len(train_rows)) if labels[j] == cn]).mean(0)
                     for cn in sorted(set(labels))}
            Pt = Hk[test_rows] @ S_class.T
            Pt_r = Hk[test_rows] @ Qrand
            for j in range(len(test_rows)):
                v, vr = Pt[j], Pt_r[j]
                best = max(mus, key=lambda cn: float(v @ mus[cn]) /
                           max(float(np.linalg.norm(mus[cn])), 1e-30))
                accs.append(1.0 if best == true_cn else 0.0)
                best_r = max(mus_r, key=lambda cn: float(vr @ mus_r[cn]) /
                             max(float(np.linalg.norm(mus_r[cn])), 1e-30))
                accs_rand.append(1.0 if best_r == true_cn else 0.0)
        acc = float(np.mean(accs))
        acc_rand = float(np.mean(accs_rand))
        out[m] = {'loeo_acc': acc, 'acc_rand': acc_rand, 'n_test_rows': len(accs),
                  'n_folds': len(usable), 'chance': 0.2}
        log('arm_a %s: LOEO acc=%.4f (chance 0.2, rand ctrl %.4f, %d rows)'
            % (m, acc, acc_rand, len(accs)))
    gate = all(out[m]['loeo_acc'] >= ARM_A_GATE for m in MODELS)
    log('arm_a gate (acc >= %.2f x3): %s' % (ARM_A_GATE, gate))
    return out, gate


# ------------------------------------------------------- arm (b)
def arm_b():
    r66 = json.load(io.open(SRC['p3166_result'], encoding='utf-8'))
    assert r66['res_sha8'] == CONTENT_SHA['p3166'][0]
    assert r66['seal_sha8'] == CONTENT_SHA['p3166'][1]
    out = {}
    for m in MODELS:
        z = np.load(SRC[MODEL_NPZ[m][0]])
        H = z['H']
        NTn, NPn, NHn, Dn = H.shape
        assert (NTn, NPn) == (3, 246), ('3151/3152 panel', H.shape)
        NL = NHn - 1
        Hk = H[:, :, NL, :].reshape(NTn * NPn, Dn).astype(np.float64)
        y = np.zeros(NTn * NPn, bool)
        ent_rows = {i: [] for i in range(NE)}
        for t in range(NTn):
            for pi, (ei, ci) in enumerate(PAIRS_FULL):
                ent_rows[ei].append(t * NPn + pi)
                if ci == CLS_OF[ei]:
                    y[t * NPn + pi] = True
        z57 = np.load(SRC[MODEL_NPZ[m][1]])
        H57 = z57['H'].astype(np.float64)
        NL57 = H57.shape[1] - 1
        X = H57[:, NL57, :]
        Xc = X - X.mean(0, keepdims=True)
        _, _, Vh57 = np.linalg.svd(Xc, full_matrices=False)
        K_ent = Vh57[:8]
        ents = sorted(ent_rows.keys())
        R8_full = build_R(Hk, ent_rows, y, ents)
        _, t1_full = kross(R8_full, K_ent)
        anchor = CENSUS_ANCHOR_RK[m]
        drift_full = abs(round(t1_full, 3) - anchor)
        assert drift_full <= 0.05, ('full-sample census drift', m, t1_full, anchor)
        angles = []
        for e in ents:
            others = [i for i in ents if i != e]
            R8_e = build_R(Hk, ent_rows, y, others)
            _, t1_e = kross(R8_e, K_ent)
            angles.append(t1_e)
        loeo_mean = float(np.mean(angles))
        band_full, band_loeo = cls_of_top1(anchor), cls_of_top1(loeo_mean)
        out[m] = {'angle_full_deg': t1_full, 'loeo_mean_deg': loeo_mean,
                  'loeo_min_deg': float(np.min(angles)), 'loeo_max_deg': float(np.max(angles)),
                  'n_folds': len(ents), 'band_full': band_full, 'band_loeo': band_loeo,
                  'census_anchor_deg': anchor, 'full_drift_deg': drift_full}
        log('arm_b %s: full=%.3f (anchor %.3f, drift %.4f) LOEO mean=%.3f '
            '[%.3f, %.3f] band %s/%s' % (m, t1_full, anchor, drift_full, loeo_mean,
                                         np.min(angles), np.max(angles), band_full, band_loeo))
    gate = all(out[m]['band_full'] == out[m]['band_loeo'] and
               abs(out[m]['loeo_mean_deg'] - out[m]['census_anchor_deg']) < ARM_B_TOL_DEG
               for m in MODELS)
    log('arm_b gate (|loeo-anchor| < %.1f deg + band agree x3): %s' % (ARM_B_TOL_DEG, gate))
    return out, gate


# ------------------------------------------------------- arm (c)
def arm_c(tidc_by_model, W_rows_by_model):
    r65 = json.load(io.open(SRC['p3165_result'], encoding='utf-8'))
    assert r65['res_sha8'] == CONTENT_SHA['p3165'][0]
    assert r65['seal_sha8'] == CONTENT_SHA['p3165'][1]
    ax_attr = json.loads(str(np.load(SRC['p2874'], allow_pickle=True)['pairs_json']))
    ax_syn = json.loads(str(np.load(SRC['p2878'], allow_pickle=True)['pairs_json']))
    ord_attr = [str(x) for x in np.load(SRC['p2874'], allow_pickle=True)['axes']]
    ord_syn = [str(x) for x in np.load(SRC['p2878'], allow_pickle=True)['axes']]
    out = {}
    for m in MODELS:
        z58 = np.load(SRC[MODEL_NPZ[m][2]])
        K_read = z58['top64'].astype(np.float64).T
        rows = W_rows_by_model[m]['rows']
        res = {}
        for tag, AX, order in (('attr', ax_attr, ord_attr), ('syntax', ax_syn, ord_syn)):
            dirs, kept_n, dropped = [], 0, []
            for a in order:
                pairs = AX[a]
                words = []
                for _, wp, wm in pairs:
                    for w in (wp, wm):
                        if w not in words:
                            words.append(w)
                kept = [w for w in words if w in rows]
                kept_n += len(kept)
                pd = []
                for _, wp, wm in pairs:
                    if wp in kept and wm in kept:
                        pd.append(unit(rows[wp] - rows[wm]))
                if len(pd) >= 2:
                    dirs.append(unit(np.stack(pd).mean(0)))
                else:
                    dropped.append(a)
            dW = np.stack(dirs) if dirs else np.zeros((0, K_read.shape[1]))
            _, t1 = kross(dW, K_read) if dW.shape[0] >= 1 else (None, float('nan'))
            res[tag] = {'top1_deg': t1, 'n_axes': int(dW.shape[0]), 'n_words': kept_n,
                        'dropped_axes': dropped}
        a4 = res['attr']['top1_deg']
        s4 = res['syntax']['top1_deg']
        out[m] = {'attr_top1_deg': a4, 'syntax_top1_deg': s4,
                  'attr_n_axes': res['attr']['n_axes'], 'syntax_n_axes': res['syntax']['n_axes'],
                  'attr_n_words': res['attr']['n_words'], 'syntax_n_words': res['syntax']['n_words'],
                  'attr_dropped': res['attr']['dropped_axes'], 'syntax_dropped': res['syntax']['dropped_axes']}
        log('arm_c %s: KxS_attr=%.3f (%d axes/%d words, dropped %s) KxS_syntax=%.3f '
            '(%d axes/%d words, dropped %s)' % (
                m, a4, res['attr']['n_axes'], res['attr']['n_words'], res['attr']['dropped_axes'],
                s4, res['syntax']['n_axes'], res['syntax']['n_words'], res['syntax']['dropped_axes']))
    d4 = abs(out['qwen3-4b']['attr_top1_deg'] - 68.094)
    d4s = abs(out['qwen3-4b']['syntax_top1_deg'] - 58.488)
    assert d4 <= ARM_C_TOL_4B and d4s <= ARM_C_TOL_4B, ('4b rebuild drift', d4, d4s)
    log('arm_c 4b rebuild vs 3165 anchors: attr drift %.4f / syntax drift %.4f deg' % (d4, d4s))
    gate = all(out[m]['attr_top1_deg'] >= 30.0 and out[m]['syntax_top1_deg'] >= 30.0
               for m in ('qwen3-14b', 'glm4-9b'))
    log('arm_c gate (14b/glm4 top1 >= 30 deg both subspaces): %s' % gate)
    return out, gate, (d4, d4s)


# ------------------------------------------------------- upgrade builders
def collect_W_rows(m):
    """One W_U load per model; keep only needed word rows (memory)."""
    mdir = os.path.join(BASE, 'models', 'hf', MODELS_HF[m])
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(mdir, local_files_only=True,
                                        trust_remote_code=True, use_fast=True)
    tidc = TidCache(tok)
    W, cfgm = load_WU(mdir)
    e2806 = json.load(io.open(SRC['p2806_exec'], encoding='utf-8'))
    CATS = e2806['cats']
    words = [w for v in CATS.values() for w in v]
    for npz_tag in ('p2874', 'p2878'):
        z = np.load(SRC[npz_tag], allow_pickle=True)
        AX = json.loads(str(z['pairs_json']))
        for a, pairs in AX.items():
            for _, wp, wm in pairs:
                words.extend([wp, wm])
    rows = {}
    for w in sorted(set(words)):
        try:
            rows[w] = W[tidc.tid(w)]
        except AssertionError:
            pass
    need_w = {w for w in words}
    return {'W': W, 'rows': rows, 'tidc': tidc, 'cfg': cfgm, 'need': need_w}


def build_upgrades(ra, ga, rb, gb, rc, gc, drift_c):
    reg12 = json.load(io.open(SRC['registry_v12'], encoding='utf-8'))
    feats = {f['id']: f for f in reg12['features']}
    upg = []

    # ---- FTR-04
    f4 = copy.deepcopy(feats['FTR-04'])
    stmt_add = (' 3176 held-out 定判（零 GPU）：S_class 方向在 3169 面板已见类行上做留一实体 '
                'LOEO 类读出（35 可映射实体 x 5 p2806 可对应类 x 3 模板 = 525 测试行，'
                'chance=0.2）：acc = %.4f/%.4f/%.4f（4b/14b/glm4），随机 10 维正交对照 '
                '%.4f/%.4f/%.4f——类词方向配方承载 held-out 实体的类身份读出，held_out 锚入表。'
                ) % (ra['qwen3-4b']['loeo_acc'], ra['qwen3-14b']['loeo_acc'], ra['glm4-9b']['loeo_acc'],
                     ra['qwen3-4b']['acc_rand'], ra['qwen3-14b']['acc_rand'], ra['glm4-9b']['acc_rand'])
    if ga:
        f4['statement'] = f4['statement'] + stmt_add
        f4['evidence_level'] = 'E2_predictive'
        f4['anchors'] = f4['anchors'] + [{
            'src': 'p3169', 'phase': 3169, 'role': 'held_out_loeo_class_readout',
            'tags': ['held_out'],
            'asserts': {
                'loeo_acc_qwen3_4b': ra['qwen3-4b']['loeo_acc'],
                'loeo_acc_qwen3_14b': ra['qwen3-14b']['loeo_acc'],
                'loeo_acc_glm4_9b': ra['glm4-9b']['loeo_acc'],
                'acc_rand_qwen3_4b': ra['qwen3-4b']['acc_rand'],
                'acc_rand_qwen3_14b': ra['qwen3-14b']['acc_rand'],
                'acc_rand_glm4_9b': ra['glm4-9b']['acc_rand'],
                'chance': 0.2, 'n_test_rows_per_model': ra['qwen3-4b']['n_test_rows'],
                'gate': ARM_A_GATE,
                'res_sha8': CONTENT_SHA['p3169'][0], 'seal_sha8': CONTENT_SHA['p3169'][1]}}]
        f4['values'] = dict(f4['values'], loeo_acc_qwen3_4b=ra['qwen3-4b']['loeo_acc'],
                            loeo_acc_qwen3_14b=ra['qwen3-14b']['loeo_acc'],
                            loeo_acc_glm4_9b=ra['glm4-9b']['loeo_acc'])
        f4['scope_limits'] = f4['scope_limits'] + ' held-out 读出限 5 可映射类（颜色无 p2806 轴）；chance=0.2。'
        upg.append({'node': 'FTR-04', 'change': 'E1_repeatable -> E2_predictive '
                    '(held_out tag; p3169 LOEO class readout 3/3 pass)',
                    'reason': '3176 arm (a): S_class directions carry held-out entity '
                              'class identity (LOEO acc %.4f/%.4f/%.4f vs chance 0.2)'
                              % (ra['qwen3-4b']['loeo_acc'], ra['qwen3-14b']['loeo_acc'],
                                 ra['glm4-9b']['loeo_acc'])})
    else:
        f4['counter_evidence'] = f4['counter_evidence'] + [
            '3176 arm (a) FAILED gate: LOEO acc %.4f/%.4f/%.4f vs 0.5 gate — held_out 升级未达成'
            % (ra['qwen3-4b']['loeo_acc'], ra['qwen3-14b']['loeo_acc'], ra['glm4-9b']['loeo_acc'])]

    # ---- FTR-06
    f6 = copy.deepcopy(feats['FTR-06'])
    stmt_add6 = (' 3176 跨模型重建定判（零 GPU，unembed-only）：AXES 词表 verbatim（2874/2878 '
                 'pairs_json），per-model W_U + single-token 重建 S_attr/S_syntax，'
                 'K_readout x S_attr = %.3f/%.3f/%.3f 度、x S_syntax = %.3f/%.3f/%.3f 度'
                 '（4b/14b/glm4）；4b 重建对拍 3165 锚 drift %.4f/%.4f 度。'
                 ) % (rc['qwen3-4b']['attr_top1_deg'], rc['qwen3-14b']['attr_top1_deg'],
                      rc['glm4-9b']['attr_top1_deg'],
                      rc['qwen3-4b']['syntax_top1_deg'], rc['qwen3-14b']['syntax_top1_deg'],
                      rc['glm4-9b']['syntax_top1_deg'], drift_c[0], drift_c[1])
    if gc:
        f6['statement'] = f6['statement'] + stmt_add6
        f6['evidence_level'] = 'E2_predictive'
        f6['model_scope'] = list(MODELS)
        f6['anchors'] = f6['anchors'] + [{
            'src': 'p2874+p2878+p3158', 'phase': 3176,
            'role': 'cross_model_unembed_rebuild', 'tags': ['cross_model'],
            'asserts': {
                'attr_top1_qwen3_4b': rc['qwen3-4b']['attr_top1_deg'],
                'attr_top1_qwen3_14b': rc['qwen3-14b']['attr_top1_deg'],
                'attr_top1_glm4_9b': rc['glm4-9b']['attr_top1_deg'],
                'syntax_top1_qwen3_4b': rc['qwen3-4b']['syntax_top1_deg'],
                'syntax_top1_qwen3_14b': rc['qwen3-14b']['syntax_top1_deg'],
                'syntax_top1_glm4_9b': rc['glm4-9b']['syntax_top1_deg'],
                'attr_n_words_4b': rc['qwen3-4b']['attr_n_words'],
                'syntax_n_words_4b': rc['qwen3-4b']['syntax_n_words'],
                'attr_n_words_14b': rc['qwen3-14b']['attr_n_words'],
                'syntax_n_words_14b': rc['qwen3-14b']['syntax_n_words'],
                'attr_n_words_glm4': rc['glm4-9b']['attr_n_words'],
                'syntax_n_words_glm4': rc['glm4-9b']['syntax_n_words'],
                'gate_deg': 30.0,
                'src_file_sha8': {'p2874': SHA_ANCHOR['p2874'], 'p2878': SHA_ANCHOR['p2878'],
                                  'p3158_4b': SHA_ANCHOR['p3158_4b'],
                                  'p3158_14b': SHA_ANCHOR['p3158_14b'],
                                  'p3158_glm4': SHA_ANCHOR['p3158_glm4']}}}]
        f6['values'] = dict(f6['values'], attr_deg_14b=rc['qwen3-14b']['attr_top1_deg'],
                            syntax_deg_14b=rc['qwen3-14b']['syntax_top1_deg'],
                            attr_deg_glm4=rc['glm4-9b']['attr_top1_deg'],
                            syntax_deg_glm4=rc['glm4-9b']['syntax_top1_deg'])
        f6['counter_evidence'] = [
            '各模型保留词数不同（单 token 集差异），轴方向为该模型 unembed 面内的重建；'
            '14b/glm4 若有轴被剔除已逐轴登记（dropped_axes）',
            '数值敏感带：S_class 行空间病态（原条保留，dtype 口径=W_U float32 链）']
        f6['scope_limits'] = ('qwen3-4b/14b/glm4 三模型（3176 unembed-only 重建后）；'
                              '矩阵链=per-model W_U（4b tie 下与 2874 embed_tokens 同矩阵）。')
        upg.append({'node': 'FTR-06', 'change': 'E1_repeatable -> E2_predictive '
                    '(cross_model; model_scope qwen3-4b -> 3 models)',
                    'reason': '3176 arm (c): unembed-only rebuild reproduces the separable '
                              'band cross-model (attr %.1f/%.1f/%.1f deg, syntax %.1f/%.1f/%.1f deg)'
                              % (rc['qwen3-4b']['attr_top1_deg'], rc['qwen3-14b']['attr_top1_deg'],
                                 rc['glm4-9b']['attr_top1_deg'], rc['qwen3-4b']['syntax_top1_deg'],
                                 rc['qwen3-14b']['syntax_top1_deg'], rc['glm4-9b']['syntax_top1_deg'])})
    else:
        f6['statement'] = f6['statement'] + stmt_add6
        f6['counter_evidence'] = f6['counter_evidence'] + [
            '3176 arm (c) FAILED gate: 14b/glm4 top1 < 30 deg — cross_model 升级未达成']

    # ---- FTR-08
    f8 = copy.deepcopy(feats['FTR-08'])
    stmt_add8 = (' 3176 held-out 定判（零 GPU）：R_logic 留一实体 LOEO（41 fold，fold 内 '
                 '40 实体重建 top8）x 固定 K_entity（3157 verbatim）：LOEO 均值角 = '
                 '%.3f/%.3f/%.3f 度 vs 全量锚 %.3f/%.3f/%.3f 度（|差| < %.0f 度门，分类带一致 '
                 '%s）——R x K_entity 混合读数对构建集实体选择稳健，held_out 锚入表。'
                 ) % (rb['qwen3-4b']['loeo_mean_deg'], rb['qwen3-14b']['loeo_mean_deg'],
                      rb['glm4-9b']['loeo_mean_deg'],
                      rb['qwen3-4b']['census_anchor_deg'], rb['qwen3-14b']['census_anchor_deg'],
                      rb['glm4-9b']['census_anchor_deg'], ARM_B_TOL_DEG,
                      '/'.join(rb[m]['band_loeo'] for m in MODELS))
    if gb:
        f8['statement'] = f8['statement'] + stmt_add8
        f8['evidence_level'] = 'E2_predictive'
        f8['anchors'] = f8['anchors'] + [{
            'src': 'p3152+p3151', 'phase': 3176,
            'role': 'held_out_loeo_rlogic_vs_kentity', 'tags': ['held_out'],
            'asserts': {
                'loeo_mean_qwen3_4b': rb['qwen3-4b']['loeo_mean_deg'],
                'loeo_mean_qwen3_14b': rb['qwen3-14b']['loeo_mean_deg'],
                'loeo_mean_glm4_9b': rb['glm4-9b']['loeo_mean_deg'],
                'angle_full_qwen3_4b': rb['qwen3-4b']['angle_full_deg'],
                'angle_full_qwen3_14b': rb['qwen3-14b']['angle_full_deg'],
                'angle_full_glm4_9b': rb['glm4-9b']['angle_full_deg'],
                'census_anchor_qwen3_4b': rb['qwen3-4b']['census_anchor_deg'],
                'census_anchor_qwen3_14b': rb['qwen3-14b']['census_anchor_deg'],
                'census_anchor_glm4_9b': rb['glm4-9b']['census_anchor_deg'],
                'tol_deg': ARM_B_TOL_DEG,
                'src_file_sha8': {'p3152_4b': SHA_ANCHOR['p3152_4b'],
                                  'p3152_14b': SHA_ANCHOR['p3152_14b'],
                                  'p3151_glm4': SHA_ANCHOR['p3151_collect'],
                                  'p3157_4b': SHA_ANCHOR['p3157_4b'],
                                  'p3157_14b': SHA_ANCHOR['p3157_14b'],
                                  'p3157_glm4': SHA_ANCHOR['p3157_glm4']}}}]
        f8['values'] = dict(f8['values'], loeo_mean_4b=rb['qwen3-4b']['loeo_mean_deg'],
                            loeo_mean_14b=rb['qwen3-14b']['loeo_mean_deg'],
                            loeo_mean_glm4=rb['glm4-9b']['loeo_mean_deg'])
        f8['scope_limits'] = f8['scope_limits'] + ' LOEO fold 的 eff 谱未逐 fold 设门（登记均值）。'
        upg.append({'node': 'FTR-08', 'change': 'E1_repeatable -> E2_predictive '
                    '(held_out tag; LOEO R_logic x K_entity stability 3/3)',
                    'reason': '3176 arm (b): R x K_entity mixing reproduces under '
                              'leave-one-entity-out (LOEO mean %.3f/%.3f/%.3f deg vs anchor)'
                              % (rb['qwen3-4b']['loeo_mean_deg'], rb['qwen3-14b']['loeo_mean_deg'],
                                 rb['glm4-9b']['loeo_mean_deg'])})
    else:
        f8['statement'] = f8['statement'] + stmt_add8
        f8['counter_evidence'] = f8['counter_evidence'] + [
            '3176 arm (b) FAILED gate: LOEO-anchor drift or band flip — held_out 升级未达成']

    return reg12, {'FTR-04': f4, 'FTR-06': f6, 'FTR-08': f8}, upg, (ga, gc, gb)


# ------------------------------------------------------- html render (3173 discipline)
CSS = """
:root { --ink:#1a1d23; --muted:#5b6472; --line:#d9dee6; --bg:#f7f8fa; --card:#ffffff;
  --k:#2456a6; --s:#0f7b6c; --r:#8a4fb8; --mech:#b35900; --ctx:#7a5c12; --ctl:#8c2f39;
  --read:#3a5a8c; --gate:#5a4a8a; --limit:#444; --sxk:#316aa8; --rxs:#2f855a; --rxk:#6b46a0; }
* { box-sizing: border-box; }
body { margin:0; padding:24px 28px 60px; background:var(--bg); color:var(--ink);
  font:14px/1.65 "Segoe UI","Microsoft YaHei","PingFang SC",sans-serif; }
h1 { font-size:22px; margin:0 0 4px; }
h2 { font-size:17px; margin:34px 0 10px; padding-bottom:6px; border-bottom:2px solid var(--line); }
h3 { font-size:14px; margin:0 0 8px; color:var(--muted); font-weight:600; }
.sub { color:var(--muted); font-size:12.5px; }
.mono { font-family:Consolas,"Courier New",monospace; font-size:12px; }
.card { background:var(--card); border:1px solid var(--line); border-radius:8px; padding:14px 16px; margin:10px 0; }
.badge { display:inline-block; border-radius:10px; padding:1px 9px; font-size:11.5px; margin-right:6px;
  border:1px solid var(--line); background:#eef1f5; color:var(--ink); }
.b-E2_predictive { background:#e3f0e3; border-color:#9fcaa0; color:#1e5c22; }
.b-E1_repeatable { background:#e5eefa; border-color:#a8c4e4; color:#24558a; }
.b-E3_causal_scoped { background:#f7e8dd; border-color:#ddb28f; color:#8a4b16; }
.b-closed, .b-closed_v1 { background:#e3f0e3; border-color:#9fcaa0; color:#1e5c22; }
.b-open { background:#fbe9e7; border-color:#e0a89f; color:#93321f; }
.b-quantified_collapse { background:#fde8e8; border-color:#e8a0a0; color:#8c1f1f; }
.b-appendix_open { background:#eee8f7; border-color:#c5b1e6; color:#5b3a99; }
.b-appendix_open_crossline { background:#f1eee2; border-color:#d4c99a; color:#6d5f1e; }
.grid { display:grid; grid-template-columns:repeat(auto-fill,minmax(340px,1fr)); gap:10px; }
table { border-collapse:collapse; width:100%; background:var(--card); border:1px solid var(--line); }
th,td { border:1px solid var(--line); padding:6px 9px; text-align:left; vertical-align:top; font-size:13px; }
th { background:#eef1f5; font-weight:600; }
.kv { margin:3px 0; }
.kv b { color:var(--muted); font-weight:600; font-size:12px; margin-right:6px; }
.anchorbox { background:#f4f6f9; border:1px dashed var(--line); border-radius:6px; padding:6px 9px;
  margin-top:8px; font-size:12px; color:#3c4550; word-break:break-all; }
.asserts { color:#52616f; }
.evid { color:#52616f; font-size:12.5px; margin:3px 0; }
.prereg { border-left:3px solid #b35900; padding-left:10px; margin-top:8px; }
.mech { border-left:3px solid #8a4fb8; padding-left:10px; margin-top:8px; white-space:pre-wrap; }
.footer { margin-top:40px; padding-top:12px; border-top:1px solid var(--line); color:var(--muted); font-size:12px; }
"""


def esc(s):
    return html.escape(str(s), quote=False)


CREATED_STR = datetime.datetime.now().strftime('%Y-%m-%d %H:%M')


def span(key, val):
    return '<span class="fv" data-k="%s">%s</span>' % (key, esc(val))


def render_feature(f):
    p = f['id']
    rows = []
    rows.append('<div class="card" id="%s">' % p)
    rows.append('<h3>%s &middot; %s <span class="badge b-%s">%s</span>'
                '<span class="badge">%s</span></h3>' % (
                    p, span(p + '.family', f['family']), f['evidence_level'],
                    span(p + '.evidence_level', f['evidence_level']),
                    esc(', '.join(f['model_scope']))))
    rows.append('<div class="kv"><b>statement</b>%s</div>' % span(p + '.statement', f['statement']))
    rows.append('<div class="kv"><b>model_scope</b>%s</div>' % span(p + '.model_scope', ', '.join(f['model_scope'])))
    if f.get('values'):
        vk = ' '.join('<span class="mono">%s=%s</span>' % (esc(k), span('%s.values.%s' % (p, k), fmt(v)))
                      for k, v in f['values'].items())
        rows.append('<div class="kv"><b>values</b>%s</div>' % vk)
    for i, a in enumerate(f['anchors']):
        rows.append('<div class="anchorbox"><b>anchor[%d]</b> src=%s phase=%s role=%s'
                    '<div class="asserts">%s</div></div>' % (
                        i, span('%s.anchors.%d.src' % (p, i), a['src']),
                        span('%s.anchors.%d.phase' % (p, i), a['phase']),
                        span('%s.anchors.%d.role' % (p, i), a['role']),
                        span('%s.anchors.%d.asserts' % (p, i),
                             json.dumps(a['asserts'], ensure_ascii=False, sort_keys=True))))
    rows.append('<div class="kv"><b>n_anchors</b>%s</div>' % span(p + '.n_anchors', str(len(f['anchors']))))
    rows.append('<div class="kv"><b>counter_evidence</b>%s</div>' % span(
        p + '.counter_evidence', ' | '.join(f['counter_evidence']) if f['counter_evidence'] else '(none)'))
    rows.append('<div class="kv"><b>replication</b>%s</div>' % span(p + '.replication', f['replication']))
    rows.append('<div class="kv"><b>scope_limits</b>%s</div>' % span(
        p + '.scope_limits', f['scope_limits'] if f['scope_limits'] else '(none)'))
    rows.append('</div>')
    return '\n'.join(rows)


def render_html(reg, nodes, gl, feats, feat_subset, node_subset, fail_subset, upg_subset,
                meta_sha):
    H = []
    H.append('<!DOCTYPE html><html lang="zh"><head><meta charset="utf-8">')
    H.append('<title>Atlas v1.4 - RDC/LPF 语义图谱</title><style>%s</style></head>' % CSS)
    H.append('<body>')
    H.append('<h1>Atlas v1.4 &mdash; 语义机制图谱（registry v1.3 渲染）</h1>')
    H.append('<div class="sub">Phase 3176 G5-B1 &middot; 零 GPU E1->E2 批量升级 &middot; registry v1.3 sha8=%s'
             ' &middot; registry v1.2 sha8=%s &middot; gap ledger v1.5 sha8=%s（3175 定稿，原样渲染）'
             ' &middot; 升级臂源 p3166=%s / p3165=%s / p3169=%s &middot; 渲染于 %s</div>' % (
                 span('meta.registry_v13_sha8', meta_sha['reg13']),
                 span('meta.registry_v12_sha8', SHA_ANCHOR['registry_v12']),
                 span('meta.gap_v15_sha8', SHA_ANCHOR['gap_v15']),
                 span('meta.p3166_res', SHA_ANCHOR['p3166_result']),
                 span('meta.p3165_res', SHA_ANCHOR['p3165_result']),
                 span('meta.p3169_res', SHA_ANCHOR['p3169_result']),
                 span('meta.created', CREATED_STR)))
    H.append('<h2>证据级 taxonomy 与三原则</h2><div class="card">')
    for k, v in reg['evidence_level_taxonomy'].items():
        H.append('<div class="kv"><b>%s</b>%s</div>' % (esc(k), esc(v)))
    H.append('<hr style="border:none;border-top:1px solid var(--line);margin:8px 0">')
    for i, pr in enumerate(reg['principles']):
        H.append('<div class="kv"><b>原则%d</b>%s</div>' % (i + 1, esc(pr)))
    H.append('</div>')

    H.append('<h2>图谱节点基座（16 节点，3162 audit disk_verified）</h2>')
    H.append('<table><tr><th>id</th><th>title</th><th>evidence</th><th>checks</th></tr>')
    for n in node_subset:
        H.append('<tr><td class="mono">%s</td><td>%s</td><td><span class="badge b-%s">%s</span>%s</td>'
                 '<td class="mono">%s</td></tr>' % (
                     n['id'], span(n['id'] + '.title', n['title']),
                     n['evidence_level'], span(n['id'] + '.elevel', n['evidence_level']),
                     span(n['id'] + '.status', n['status']),
                     span(n['id'] + '.checks', '%d/%d' % (n['n_pass'], n['n_checks']))))
    H.append('</table>')

    fams = {}
    for f in feat_subset:
        fams.setdefault(f['family'], []).append(f)
    H.append('<h2>跨模型稳定特征登记表（registry v1.3，%d 条）</h2>' % len(reg['features']))
    for fam in sorted(fams):
        H.append('<h3>family = %s（%d 条）</h3>' % (esc(fam), len(fams[fam])))
        H.append('<div class="grid">')
        for f in fams[fam]:
            H.append(render_feature(f))
        H.append('</div>')

    H.append('<h2>E 级 / 范围升级链</h2><div class="card">')
    for i, u in upg_subset:
        H.append('<div class="kv"><b>#%d</b>%s &nbsp;%s</div><div class="evid">reason: %s</div>' % (
            i + 1, span('UPG.%d.node' % i, u['node']), span('UPG.%d.change' % i, u['change']),
            span('UPG.%d.reason' % i, u['reason'])))
    H.append('</div>')

    H.append('<h2>失败账本（F1-F%d，可证伪性存档）</h2>' % len(reg['failures']))
    H.append('<table><tr><th>id</th><th>kind</th><th>phase</th><th>text</th><th>evidence</th></tr>')
    for fl in fail_subset:
        H.append('<tr><td class="mono">%s</td><td>%s</td><td class="mono">%s</td><td>%s</td><td class="mono">%s</td></tr>' % (
            fl['id'], span(fl['id'] + '.kind', fl['kind']), span(fl['id'] + '.phase', fl['phase']),
            span(fl['id'] + '.text', fl['text']), span(fl['id'] + '.evidence', fl['evidence'])))
    H.append('</table>')

    H.append('<h2>缺口账本 v1.5（GAP-4 quantified_collapse 定稿：3171 机制定位 + 3172 干预 + 3175 残留定判）</h2>')
    for g in gl['gaps']:
        closed_at = str(g['closed_at_phase']) if g.get('closed_at_phase') is not None else '-'
        badge_closed = '<span class="badge">closed@%s</span>' % span(g['id'] + '.closed_at', closed_at) \
            if g['status'].startswith('closed') else ''
        H.append('<div class="card"><h3>%s &middot; %s <span class="badge b-%s">%s</span>%s</h3>' % (
            g['id'], span(g['id'] + '.title', g['title']), g['status'], span(g['id'] + '.status', g['status']), badge_closed))
        H.append('<div class="kv"><b>statement</b>%s</div>' % span(g['id'] + '.statement', g['statement']))
        H.append('<div class="evid">evidence:</div>')
        H.append('<div class="kv">%s</div>' % span(g['id'] + '.evidence', ' || '.join(g['evidence'])))
        H.append('<div class="anchorbox"><b>anchors</b> %s</div>' % span(
            g['id'] + '.anchors', ', '.join('%s=%s' % (k, v) for k, v in sorted(g['anchor_sha8'].items()))))
        if g['id'] == 'GAP-4':
            pr = g['prereg']
            H.append('<div class="prereg"><div class="kv"><b>prereg</b>Phase %s %s（%s）</div>' % (
                span('GAP-4.prereg.phase', pr['phase']), span('GAP-4.prereg.name', pr['name']),
                span('GAP-4.prereg.status', pr['status'])))
            H.append('<div class="kv"><b>hypothesis</b>%s</div>' % span('GAP-4.prereg.hypothesis', pr['hypothesis']))
            H.append('<div class="kv"><b>protocol</b>%s</div>' % span('GAP-4.prereg.protocol', pr['protocol']))
            H.append('<div class="kv"><b>gate</b>%s</div></div>' % span('GAP-4.prereg.gate', pr['gate']))
            H.append('<div class="mech"><div class="kv"><b>mechanism_note</b>%s</div></div>' % span(
                'GAP-4.mechanism_note', g['mechanism_note']))
        H.append('</div>')

    H.append('<h2>附录挂账（%d 项，逐条证据锚）</h2><div class="card">' % len(gl['appendix']))
    for a in gl['appendix']:
        H.append('<div class="kv"><b>%s</b>%s <span class="badge b-%s">%s</span>'
                 '<span class="mono">anchor=%s(%s)</span><div class="evid">%s</div></div>' % (
                     a['id'], span(a['id'] + '.title', a['title']), a['status'],
                     span(a['id'] + '.status', a['status']),
                     span(a['id'] + '.anchor', a['anchor']),
                     span(a['id'] + '.anchor_sha8', a['anchor_sha8']),
                     span(a['id'] + '.note', a['note'])))
    H.append('</div>')

    H.append('<div class="footer">RDC/LPF atlas v1.4 &middot; 渲染自 atlas_registry_v1_3.json (%s)'
             ' + gap ledger v1.5 (%s，3175 定稿原样渲染) &middot; 增量来源: Phase 3176 G5-B1 三臂'
             '（p3166 %s / p3165 %s / p3169 %s）&middot; 本文件为 Phase 3176 封存产物，字段由 data-k 校验器逐字段对盘。</div>' % (
                 meta_sha['reg13'], SHA_ANCHOR['gap_v15'],
                 SHA_ANCHOR['p3166_result'], SHA_ANCHOR['p3165_result'], SHA_ANCHOR['p3169_result']))
    H.append('</body></html>')
    return '\n'.join(H)


def flatten(reg, nodes, gl):
    d = {}
    for f in reg['features']:
        p = f['id']
        d[p + '.statement'] = f['statement']
        d[p + '.family'] = f['family']
        d[p + '.evidence_level'] = f['evidence_level']
        d[p + '.model_scope'] = ', '.join(f['model_scope'])
        d[p + '.n_anchors'] = str(len(f['anchors']))
        d[p + '.counter_evidence'] = ' | '.join(f['counter_evidence']) if f['counter_evidence'] else '(none)'
        d[p + '.replication'] = f['replication']
        d[p + '.scope_limits'] = f['scope_limits'] if f['scope_limits'] else '(none)'
        for i, a in enumerate(f['anchors']):
            d['%s.anchors.%d.src' % (p, i)] = str(a['src'])
            d['%s.anchors.%d.phase' % (p, i)] = str(a['phase'])
            d['%s.anchors.%d.role' % (p, i)] = a['role']
            d['%s.anchors.%d.asserts' % (p, i)] = json.dumps(a['asserts'], ensure_ascii=False, sort_keys=True)
        for k, v in (f.get('values') or {}).items():
            d['%s.values.%s' % (p, k)] = fmt(v)
    for n in nodes:
        d[n['id'] + '.title'] = n['title']
        d[n['id'] + '.elevel'] = n['evidence_level']
        d[n['id'] + '.status'] = n['status']
        d[n['id'] + '.checks'] = '%d/%d' % (n['n_pass'], n['n_checks'])
    for i, u in enumerate(reg['upgrade_log']):
        d['UPG.%d.node' % i] = u['node']
        d['UPG.%d.change' % i] = u['change']
        d['UPG.%d.reason' % i] = u['reason']
    for fl in reg['failures']:
        d['%s.kind' % fl['id']] = fl['kind']
        d['%s.phase' % fl['id']] = str(fl['phase'])
        d['%s.text' % fl['id']] = fl['text']
        d['%s.evidence' % fl['id']] = str(fl['evidence'])
    for g in gl['gaps']:
        p = g['id']
        d[p + '.title'] = g['title']
        d[p + '.status'] = g['status']
        if g['status'].startswith('closed'):
            d[p + '.closed_at'] = str(g['closed_at_phase'])
        d[p + '.statement'] = g['statement']
        d[p + '.evidence'] = ' || '.join(g['evidence'])
        d[p + '.anchors'] = ', '.join('%s=%s' % (k, v) for k, v in sorted(g['anchor_sha8'].items()))
        if p == 'GAP-4':
            pr = g['prereg']
            d[p + '.prereg.phase'] = str(pr['phase'])
            d[p + '.prereg.name'] = pr['name']
            d[p + '.prereg.hypothesis'] = pr['hypothesis']
            d[p + '.prereg.protocol'] = pr['protocol']
            d[p + '.prereg.gate'] = pr['gate']
            d[p + '.prereg.status'] = pr['status']
            d[p + '.mechanism_note'] = g['mechanism_note']
    for a in gl['appendix']:
        p = a['id']
        d[p + '.title'] = a['title']
        d[p + '.status'] = a['status']
        d[p + '.note'] = a['note']
        d[p + '.anchor'] = a['anchor']
        d[p + '.anchor_sha8'] = a['anchor_sha8']
    d['meta.registry_v13_sha8'] = meta_sha_global['reg13']
    d['meta.registry_v12_sha8'] = SHA_ANCHOR['registry_v12']
    d['meta.gap_v15_sha8'] = SHA_ANCHOR['gap_v15']
    d['meta.p3166_res'] = SHA_ANCHOR['p3166_result']
    d['meta.p3165_res'] = SHA_ANCHOR['p3165_result']
    d['meta.p3169_res'] = SHA_ANCHOR['p3169_result']
    d['meta.created'] = CREATED_STR
    return d


meta_sha_global = {'reg13': None}


def verify_html(html_text, expected):
    pairs = re.findall(r'data-k="([^"]+)"[^>]*>(.*?)</span>', html_text, re.S)
    found = {}
    dups = []
    for k, v in pairs:
        if k in found:
            dups.append(k)
        found[k] = html.unescape(v)
    missing = sorted(set(expected) - set(found))
    extra = sorted(set(found) - set(expected))
    mismatch = []
    for k in sorted(set(expected) & set(found)):
        if found[k] != expected[k]:
            mismatch.append((k, found[k][:80], expected[k][:80]))
    return {'n_expected': len(expected), 'n_found': len(found), 'dups': dups,
            'missing': missing, 'extra': extra, 'mismatch': mismatch}


# ---------------------------------------------------------------- main
def main():
    d8 = freeze()
    log('== Phase 3176 g5b1_elev_upgrade %s ==' % ('SMOKE' if SMOKE else 'FULL'))

    anchors = {}
    for k, p in SRC.items():
        b = open(p, 'rb').read()
        h = sha8(b)
        assert h == SHA_ANCHOR[k], ('source sha mismatch', k, h, SHA_ANCHOR[k])
        anchors[k] = {'path': p, 'sha8': h}
    log('sources: %d files sha-asserted' % len(SRC))

    reg12 = json.loads(open(SRC['registry_v12'], 'rb').read().decode('utf-8'))
    gl15 = json.loads(open(SRC['gap_v15'], 'rb').read().decode('utf-8'))
    reg2 = json.load(io.open(os.path.join(SRC_DIR, r'phase3162\g5a1_atlas_foundation\atlas_registry.json'),
                             encoding='utf-8'))
    nodes = reg2['audit']['nodes']
    assert len(reg12['features']) == 22 and len(nodes) == 16
    assert len(reg12['failures']) == 14 and len(reg12['upgrade_log']) == 3
    assert reg12['version'] == '1.2' and gl15['version'] == '1.5'
    log('registry v1.2 (22 feats / 14 fails / 3 upgs) + gap v1.5 loaded, sha-anchored')

    # ---- arms (always FULL, zero GPU)
    log('== arm (a) FTR-04 held_out: S_class LOEO class readout ==')
    W_rows = {}
    tidc_map = {}
    for m in MODELS:
        W_rows[m] = collect_W_rows(m)
        tidc_map[m] = W_rows[m]['tidc']
    ra, ga = arm_a(tidc_map, W_rows)
    del W_rows, tidc_map
    log('== arm (b) FTR-08 held_out: R_logic LOEO x K_entity ==')
    rb, gb = arm_b()
    log('== arm (c) FTR-06 cross_model: unembed-only rebuild ==')
    W_rows = {}
    tidc_map = {}
    for m in MODELS:
        W_rows[m] = collect_W_rows(m)
        tidc_map[m] = W_rows[m]['tidc']
    rc, gc, drift_c = arm_c(tidc_map, W_rows)
    del W_rows, tidc_map

    reg13, upgraded, upg_events, gates = build_upgrades(ra, ga, rb, gb, rc, gc, drift_c)
    ga_, gc_, gb_ = gates
    n_upg = len(upg_events)
    log('arm verdicts: FTR-04 %s | FTR-06 %s | FTR-08 %s -> %d upgrade events'
        % (ga_, gc_, gb_, n_upg))

    # registry v1.3 assembly with byte-for-byte preservation checks
    v12_feats = {f['id']: f for f in reg12['features']}
    reg13['version'] = '1.3'
    reg13['supersedes'] = 'atlas_registry_v1_2.json (%s, Phase 3173)' % SHA_ANCHOR['registry_v12']
    new_feats = []
    for f in reg12['features']:
        if f['id'] in upgraded:
            new_feats.append(upgraded[f['id']])
        else:
            new_feats.append(f)
    reg13['features'] = new_feats
    assert len(reg13['features']) == 22
    for f in reg12['features']:
        fid = f['id']
        a = json.dumps(f, ensure_ascii=False, sort_keys=True)
        b_obj = [x for x in reg13['features'] if x['id'] == fid][0]
        b = json.dumps(b_obj, ensure_ascii=False, sort_keys=True)
        happened = ((fid == 'FTR-04' and ga_) or (fid == 'FTR-06' and gc_)
                    or (fid == 'FTR-08' and gb_))
        if happened:
            assert a != b, ('upgrade did not change feature', fid)
        elif fid in ('FTR-04', 'FTR-06', 'FTR-08'):
            # failed arm: only counter_evidence may gain the failure entry
            fa = json.loads(a)
            fb = json.loads(b)
            ce_old = fa.pop('counter_evidence')
            ce_new = fb.pop('counter_evidence')
            assert json.dumps(fa, ensure_ascii=False, sort_keys=True) == \
                json.dumps(fb, ensure_ascii=False, sort_keys=True), \
                ('G3 non-counter fields changed on failed arm', fid)
            assert ce_new[:len(ce_old)] == ce_old, ('G3 counter_evidence rewritten', fid)
            assert len(ce_new) == len(ce_old) + 1, ('G3 counter_evidence append', fid)
        else:
            assert a == b, ('G3 feature not preserved', fid)
    upgraded_ids = [fid for fid, ok in (('FTR-04', ga_), ('FTR-06', gc_), ('FTR-08', gb_)) if ok]
    failed_ids = [fid for fid, ok in (('FTR-04', ga_), ('FTR-06', gc_), ('FTR-08', gb_)) if not ok]
    log('G3 preservation: %d unchanged byte-for-byte; upgraded: %s; failed-arm counter append: %s'
        % (22 - len(upgraded_ids) - len(failed_ids), ','.join(upgraded_ids) or 'none',
           ','.join(failed_ids) or 'none'))

    # failures: F15 per failed arm
    fails = list(reg12['failures'])
    f15 = None
    if not (ga_ and gc_ and gb_):
        parts = []
        if not ga_:
            parts.append('FTR-04 arm (a) LOEO acc %.4f/%.4f/%.4f vs 0.5'
                         % (ra['qwen3-4b']['loeo_acc'], ra['qwen3-14b']['loeo_acc'], ra['glm4-9b']['loeo_acc']))
        if not gc_:
            parts.append('FTR-06 arm (c) 14b/glm4 attr %.3f/%.3f syntax %.3f/%.3f vs 30'
                         % (rc['qwen3-14b']['attr_top1_deg'], rc['glm4-9b']['attr_top1_deg'],
                            rc['qwen3-14b']['syntax_top1_deg'], rc['glm4-9b']['syntax_top1_deg']))
        if not gb_:
            parts.append('FTR-08 arm (b) LOEO-anchor drift/band flip')
        f15 = {'id': 'F15', 'kind': 'upgrade_gate_failed', 'phase': 3176,
               'text': 'E1->E2 升级门未全过：%s。相应特征保持 E1_repeatable，'
                       'held_out/cross_model 锚不入表（诚实登记）。' % '；'.join(parts),
               'evidence': 'p3176 run (this phase); arms a/b/c verdicts %s/%s/%s' % (ga_, gc_, gb_)}
        fails = fails + [f15]
    reg13['failures'] = fails

    # upgrade_log: first 3 byte-for-byte, append events
    upgs = list(reg12['upgrade_log'])
    upgs = upgs + upg_events
    reg13['upgrade_log'] = upgs
    for i in range(3):
        a = json.dumps(reg12['upgrade_log'][i], ensure_ascii=False, sort_keys=True)
        b = json.dumps(reg13['upgrade_log'][i], ensure_ascii=False, sort_keys=True)
        assert a == b, ('G7 old upgrade event changed', i)
    assert len(reg13['upgrade_log']) == 3 + n_upg
    log('G7 upgrade_log %d -> %d (first 3 byte-for-byte)' % (3, len(upgs)))

    # G4 upgrade asserts
    for fid, ok in (('FTR-04', ga_), ('FTR-06', gc_), ('FTR-08', gb_)):
        f = [x for x in reg13['features'] if x['id'] == fid][0]
        if ok:
            assert f['evidence_level'] == 'E2_predictive', ('G4 level', fid)
            tags = [t for a in f['anchors'] for t in a.get('tags', [])]
            need = 'held_out' if fid in ('FTR-04', 'FTR-08') else 'cross_model'
            assert need in tags, ('G4 tag missing', fid)
            if fid == 'FTR-06':
                assert f['model_scope'] == list(MODELS), ('G4 scope', fid)
        else:
            assert f['evidence_level'] == 'E1_repeatable', ('G4 fallback level', fid)
    log('G4 upgrade asserts OK (levels/tags/scope consistent with arm verdicts)')

    # write registry v1.3
    suffix = 'smoke_' if SMOKE else ''
    reg13_path = os.path.join(OUTDIR, suffix + 'atlas_registry_v1_3.json')
    with io.open(reg13_path, 'w', encoding='utf-8') as f:
        json.dump(reg13, f, ensure_ascii=False, indent=1)
    meta_sha_global['reg13'] = sha8_file(reg13_path)
    log('registry v1.3 written sha8 %s' % meta_sha_global['reg13'])

    # subsets
    feats_all = reg13['features']
    feats = feats_all
    fails_sub = reg13['failures']
    upgs_sub = list(enumerate(reg13['upgrade_log']))
    nodes_sub = nodes[:]
    if SMOKE:
        keep_ids = ('FTR-04', 'FTR-08', 'FTR-22')
        feats = [f for f in feats_all if f['id'] in keep_ids]
        nodes_sub = nodes[:4]
        fails_sub = fails_sub[:2]

    flat = flatten(dict(reg13, features=feats_all, failures=reg13['failures'],
                        upgrade_log=reg13['upgrade_log']),
                   nodes_sub, gl15)
    if SMOKE:
        keep = set()
        for f in feats:
            keep |= set(k for k in flat if k.startswith(f['id'] + '.'))
        for n in nodes_sub:
            keep |= set(k for k in flat if k.startswith(n['id'] + '.'))
        for i, u in upgs_sub:
            keep |= set(k for k in flat if k.startswith('UPG.%d.' % i))
        for fl in fails_sub:
            keep |= set(k for k in flat if k.startswith(fl['id'] + '.'))
        keep |= set(k for k in flat if k.startswith(('GAP-', 'APPX-', 'meta.')))
        flat = {k: v for k, v in flat.items() if k in keep}

    html_text = render_html(reg13, nodes_sub, gl15, feats, feats, nodes_sub, fails_sub,
                            upgs_sub, meta_sha_global)

    assert '<link' not in html_text and '<script' not in html_text, 'external resource found'
    assert 'http://' not in html_text and 'https://' not in html_text, 'external url found'
    log('G6 no-external OK')

    v = verify_html(html_text, flat)
    log('G1 field check: expected=%d found=%d missing=%d extra=%d mismatch=%d dups=%d' % (
        v['n_expected'], v['n_found'], len(v['missing']), len(v['extra']), len(v['mismatch']), len(v['dups'])))
    assert not v['missing'], ('missing keys', v['missing'][:10])
    assert not v['extra'], ('extra keys', v['extra'][:10])
    assert not v['mismatch'], ('mismatch', v['mismatch'][:5])
    assert not v['dups'], ('dup keys', v['dups'][:5])

    html_path = os.path.join(OUTDIR, suffix + 'atlas_v1_4.html')
    with io.open(html_path, 'w', encoding='utf-8') as f:
        f.write(html_text)
    html_sha = sha8_file(html_path)
    log('html written %s (%d B, sha8 %s)' % (os.path.basename(html_path),
                                             len(html_text.encode('utf-8')), html_sha))

    verdict = ('g5b1_elev_upgrade|arms_%s_%s_%s|upgrades_%d|features_22|failures_%d|'
               'v12_preserved_19|html_fields_%d_ok|gap_v15_rendered_as_is') % (
        'pass' if ga_ else 'fail', 'pass' if gc_ else 'fail', 'pass' if gb_ else 'fail',
        n_upg, len(reg13['failures']), v['n_found'])

    summary = {
        'phase': 3176, 'name': 'g5b1_elev_upgrade', 'smoke': SMOKE,
        'design_sha8': d8, 'verdict': verdict,
        'registry_v13_file': os.path.basename(reg13_path), 'registry_v13_sha8': meta_sha_global['reg13'],
        'html_file': os.path.basename(html_path), 'html_sha8': html_sha,
        'field_check': {'n_expected': v['n_expected'], 'n_found': v['n_found'],
                        'missing': len(v['missing']), 'extra': len(v['extra']),
                        'mismatch': len(v['mismatch']), 'dups': len(v['dups'])},
        'arm_a': {'gate': ga_, 'per_model': ra},
        'arm_b': {'gate': gb_, 'per_model': rb},
        'arm_c': {'gate': gc_, 'per_model': rc, 'drift_4b_deg': list(drift_c)},
        'upgrades': upg_events, 'n_upgrades': n_upg,
        'failure_added': f15,
        'counts': {'features': len(feats_all), 'nodes': len(nodes_sub),
                   'failures': len(reg13['failures']), 'upgrades': len(reg13['upgrade_log']),
                   'gaps': len(gl15['gaps']), 'appendix': len(gl15['appendix'])},
        'gap_status': {g['id']: g['status'] for g in gl15['gaps']},
        'sources': {k: a['sha8'] for k, a in anchors.items()},
    }
    raw = json.dumps(summary, ensure_ascii=False, indent=1, sort_keys=True)
    res8 = sha8(raw.encode('utf-8'))
    mid = json.dumps(dict(summary, res_sha8=res8), ensure_ascii=False, indent=1, sort_keys=True)
    seal8 = sha8(mid.encode('utf-8'))
    rpath = os.path.join(OUTDIR, suffix + 'result.json')
    with io.open(rpath, 'w', encoding='utf-8') as f:
        f.write(json.dumps(dict(summary, res_sha8=res8, seal_sha8=seal8),
                           ensure_ascii=False, indent=1, sort_keys=True))
    log('result written res=%s seal=%s' % (res8, seal8))
    log('verdict: %s' % verdict)

    with io.open(os.path.join(OUTDIR, suffix + 'run_log.txt'), 'w', encoding='utf-8') as f:
        f.write('\n'.join(LOG) + '\n')


if __name__ == '__main__':
    main()
