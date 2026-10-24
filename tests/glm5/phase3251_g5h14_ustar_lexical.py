# -*- coding: utf-8 -*-
"""Phase 3251 G5-H14: u* class-level lexical attribution (formalization of the
3190 downgraded arm (b)) + fidelity-closure summary (doc arm, in closeout). Zero GPU.

Prereg (frozen before any observation): MEMO Phase 3250 tail arms.
Background: the D-line quintet (3193-3197) settled the DIRECTION ALGEBRA of the
shared OOV collapse direction u*: SIGNED_CLASS_CONTRAST (u* ~= -sum w_c v_c,
w>0, vc_l1_share ~0.9). Still open since 3190: the LEXICAL arm - is u* aligned
with the 10 class first-token embedding difference directions (vocabulary-side
class structure)? 3190 downgraded this arm (weight loading outside its zero-GPU
budget); 3232 preregistered its formalization; deferred since. This phase
executes it verbatim:

  Chain replication (verbatim 3190): 3169 collect npz -> B-protocol ridge ->
  876 OOV row residual directions (unit, mean over seeds 7/8/9) -> Qn ->
  ustar = Vt[0].  Anchors: 3169 per-seed E, 3189 scalars (top1/top2/cross_med/
  within/r80/r90), 3190 sealed (loo_top1/cos_ustar_vc/ks_seen_ustar/
  loo_max_delta) - all drift == 0 bitwise.

  New lexical arm (per model, own embedding space):
    E_c = embed_tokens row of the first subword token of class word c
          (mean pooling over subwords if multi-token, 3241 correction style;
          class words are single-token under qwen3-4b tokenizer - asserted
          for 4b, recorded for others)
    grad_cls_c = unit(E_c - mean over the 10 class rows)   (mean-centered
          difference direction; pure weight-side info, declaration untouched)
    per-class SIGNED cos(ustar, grad_cls_c) and |cos| spectrum
    structure reading: oov4_sum|cos| vs seen6_sum|cos|  (u* is an OOV-residual
          top-1, so OOV-class lexical alignment is predicted to dominate)
    sign check (3197 link): count of classes with cos < 0 (anti-class-mean
          contrast predicts negative-signed alignment for carrier classes)
    controls: (a) random-direction baseline - median |cos| over 500 unit
          Gaussian directions in the model space; (b) pseudo-class word
          permutation - shuffle the 73 ENTITY words into 10 pseudo-classes
          (fixed rng 42, 20 draws), build pseudo-class mean rows and their
          mean-centered difference directions, report the null of
          max_c |cos(ustar, grad_pseudo_c)|. SMOKE-stage correction: the
          originally planned class-ROW permutation is degenerate - the
          mean-centered difference direction SET is invariant under row
          permutation (center unchanged), so its null is identically the
          observed max (same degeneracy family as the 3242 class-level
          permutation); the pseudo-class control instead tests whether the
          CLASS-WORD identity (not generic word-cluster geometry) carries
          the alignment.

  Gate (pinned pre-observation): LEXICALLY_STRUCTURED iff per model
    max_c |cos(ustar, grad_cls_c)| >= 3 x random-baseline median  AND
    oov4_sum|cos| > seen6_sum|cos|, in ALL 3 models => ustar_lexical_registered
    else => ustar_lexical_partial (characterization recorded either way).
  Report-only for the atlas: registry v1.33 byte-untouched (characterization).

SMOKE: P3251_SMOKE=1 (model qwen3-4b only, 100 random directions, 5 perm
draws, smoke_* files).
"""
import json
import os
import time
import hashlib
import numpy as np

T0 = time.time()
NAME = 'g5h14_ustar_lexical_attribution'
SMOKE = os.environ.get('P3251_SMOKE') == '1'
SRC_DIR = r'D:\AI2050\Ai2050-OpenOne\tests\glm5\result\rdc_query_construction_20260913'
ROOT = r'D:\AI2050\Ai2050-OpenOne'
OUTDIR = os.path.join(SRC_DIR, 'phase3251', NAME)
LOG = []

MODELS = ['qwen3-4b', 'qwen3-14b', 'glm4-9b']
HF_DIR = {'qwen3-4b': 'qwen3-4b', 'qwen3-14b': 'Qwen3-14B',
          'glm4-9b': 'glm4-9b-chat-hf'}
MODELS_RUN = ['qwen3-4b'] if SMOKE else MODELS
N_RAND = 100 if SMOKE else 500
N_PERM = 5 if SMOKE else 20
GATE_RAND_MULT = 3.0

NPZ69_KEY = {'qwen3-4b': 'p3169_4b', 'qwen3-14b': 'p3169_14b',
             'glm4-9b': 'p3169_glm4'}

SRC = {
    'p3169_result': os.path.join(SRC_DIR, r'phase3169\g5a6_oov_panel\result.json'),
    'p3169_4b': os.path.join(SRC_DIR, r'phase3169\g5a6_oov_panel\collect_qwen3-4b.npz'),
    'p3169_14b': os.path.join(SRC_DIR, r'phase3169\g5a6_oov_panel\collect_qwen3-14b.npz'),
    'p3169_glm4': os.path.join(SRC_DIR, r'phase3169\g5a6_oov_panel\collect_glm4-9b.npz'),
    'p3189_result': os.path.join(SRC_DIR, r'phase3189\g5c2_residual_structure\result.json'),
    'p3190_result': os.path.join(SRC_DIR, r'phase3190\g5c3_shared_direction_robustness\result.json'),
    'registry_v133': os.path.join(SRC_DIR, r'phase3237\g5g9_gap5_seal\atlas_registry_v1_33.json'),
    'p3250_result': os.path.join(SRC_DIR, r'phase3250\g5h13_kernel_boundary\result.json'),
    'ledger': os.path.join(ROOT, r'research\gpt5\atlas\atlas_ledger.json'),
}
SHA_ANCHOR = {
    'p3169_result': '5b51c2c1', 'p3169_4b': '36eb4ff0', 'p3169_14b': '97f98575',
    'p3169_glm4': '9cc3f8b4', 'p3189_result': '5bd804d0',
}

CLASSES_SEEN = ['水果', '动物', '交通工具', '家具', '金属', '颜色']
ENT_SEEN = {
    '水果': ['苹果', '香蕉', '梨', '西瓜', '葡萄', '草莓', '芒果', '柠檬'],
    '动物': ['狗', '猫', '老虎', '大象', '兔子', '猴子', '马', '牛'],
    '交通工具': ['汽车', '火车', '飞机', '摩托车', '卡车', '地铁'],
    '家具': ['桌子', '椅子', '床', '沙发', '地毯', '窗帘'],
    '金属': ['铁', '铜', '铝', '金', '银', '锌', '铅'],
    '颜色': ['红', '蓝', '绿', '黄', '黑', '白'],
}
TPL = {0: '{e}是一种{c}。', 1: '{e}属于{c}这一类。', 2: '{e}，一种常见的{c}。'}
CLASSES_OOV = ['乐器', '天气', '运动', '电器']
ENT_OOV_PANEL = {
    '乐器': ['钢琴', '小提琴', '吉他', '鼓', '笛子', '二胡', '琵琶', '口琴'],
    '天气': ['雨', '雪', '雷', '雾', '冰雹', '台风', '露水', '霜'],
    '运动': ['足球', '篮球', '乒乓球', '游泳', '跑步', '体操', '拳击', '围棋'],
    '电器': ['电视', '冰箱', '洗衣机', '空调', '微波炉', '电饭煲', '吸尘器', '风扇'],
}
NT = 3
SEEDS_S1 = [7, 8, 9]
FRAC_S1 = 0.2
LAM = 1e-3
REPL_TOL = 1e-9
CLASSES = CLASSES_SEEN + CLASSES_OOV
ENT = dict(ENT_SEEN)
ENT.update(ENT_OOV_PANEL)
ENTS = [e for cl in CLASSES for e in ENT[cl]]
NE = len(ENTS)
NC = len(CLASSES)
PAIRS = [(i, c) for i in range(NE) for c in range(NC)]
NP_ = len(PAIRS)
assert (NE, NC, NP_) == (73, 10, 730)
PI_OF_PAIR = {p: j for j, p in enumerate(PAIRS)}
ALL_OOV_ROWS = [t * NP_ + pi for t in range(NT) for pi in range(NP_)
                if PAIRS[pi][1] >= len(CLASSES_SEEN)]
assert len(ALL_OOV_ROWS) == 876
NCLS_SEEN = len(CLASSES_SEEN)

DESIGN = {
    'phase': '3251', 'name': NAME,
    'prereg': 'AGI_GPT5_MEMO Phase 3250 tail arms (b) + (a), frozen before any '
              'observation; formalization of the 3190 downgraded lexical arm '
              '(b) (3232 prereg chain, deferred to here)',
    'data': 'sealed 3169 collect npz x3 + sealed 3169/3189/3190 result.json + '
            'per-model embed_tokens first shard (weight-side only); zero GPU',
    'residual_chain': 'verbatim 3190: 3169 B-protocol (seeds 7/8/9, '
                      'split_s1_246, ridge lam 1e-3, cols 780), 876 OOV rows, '
                      'row direction = unit(mean over seeds) -> Qn; '
                      'ustar = SVD top-1 right vector Vt[0]',
    'lexical_arm': 'per model: E_c = first-subword embed row of class word '
                   '(mean pooling if multi-token; 4b single-token asserted); '
                   'grad_cls_c = unit(E_c - mean over 10 rows); per-class '
                   'signed cos(ustar, grad_cls_c); oov4 vs seen6 |cos| sums; '
                   'sign count (3197 anti-class-mean link); controls = 500 '
                   'random unit directions (median |cos|) + 20 class-row '
                   'permutations (max |cos| null)',
    'gate': 'LEXICALLY_STRUCTURED iff per model max_c |cos(ustar,grad_cls_c)| '
            '>= %g x random median AND oov4_sum > seen6_sum, ALL 3 models => '
            'ustar_lexical_registered; else ustar_lexical_partial '
            '(characterization either way; registry v1.33 byte-untouched)'
            % GATE_RAND_MULT,
    'doc_arm': '(a) fidelity-closure summary paragraph 3239-3250 + gap '
               'appendix candidate entry, written at closeout',
    'sources': SHA_ANCHOR,
}


def log(s):
    ln = '[%7.1f] %s' % (time.time() - T0, s)
    LOG.append(ln)
    try:
        print(ln, flush=True)
    except Exception:
        pass


def sha8(b):
    return hashlib.sha256(b).hexdigest()[:8]


def sha8_file(p):
    return sha8(open(p, 'rb').read())


def freeze():
    eblob = json.dumps(DESIGN, ensure_ascii=False, sort_keys=True, indent=1).encode('utf-8')
    dsha = hashlib.sha256(eblob).hexdigest()
    os.makedirs(OUTDIR, exist_ok=True)
    exe_p = os.path.join(OUTDIR, 'execution.json')
    if os.path.exists(exe_p) and not SMOKE:
        prev = json.load(open(exe_p, encoding='utf-8'))
        assert prev['design_sha8'] == dsha[:8], 'DESIGN DRIFT'
        log('freeze: existing execution.json OK (%s)' % dsha[:8])
    else:
        json.dump({'phase': 3251, 'name': NAME, 'design_sha8': dsha[:8],
                   'design': DESIGN, 'frozen_before': 'any observation',
                   'created': time.strftime('%Y-%m-%d %H:%M:%S')},
                  open(exe_p, 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
        log('freeze: execution.json written (%s)' % dsha[:8])
    return dsha[:8]


def unit(v):
    return v / (float(np.linalg.norm(v)) + 1e-18)


def split_s1_246(seed):
    SEEN_PAIR_SET = set((i, c) for i in range(41) for c in range(6))
    pairs246 = sorted(SEEN_PAIR_SET)
    rng = np.random.RandomState(seed)
    idx = rng.permutation(len(pairs246))
    n_test = int(round(FRAC_S1 * len(pairs246)))
    test = set(pairs246[j] for j in idx[:n_test])
    return SEEN_PAIR_SET - test, test


def ridge_primal(Xtr, Ytr, lam=LAM):
    A = Xtr.T @ Xtr + lam * np.eye(Xtr.shape[1], dtype=np.float32)
    W = np.linalg.solve(A, Xtr.T @ Ytr)
    return W


def phi_cols(ne, nc, nt):
    return ne + nc + nt + 1


def phi_row(i, c, t, ne, nc, nt):
    v = np.zeros(phi_cols(ne, nc, nt), np.float32)
    v[i] = 1.0
    v[ne + c] = 1.0
    v[ne + nc + t] = 1.0
    v[-1] = 1.0
    return v


def embed_shard_path(model):
    hf = os.path.join(ROOT, 'models', 'hf', HF_DIR[model])
    idx_p = os.path.join(hf, 'model.safetensors.index.json')
    if os.path.exists(idx_p):
        idx = json.load(open(idx_p, encoding='utf-8'))
        return os.path.join(hf, idx['weight_map']['model.embed_tokens.weight'])
    raise FileNotFoundError(idx_p)


_EMB = {}


def class_embed_rows(model, words):
    """first-subword embed rows (mean pooling if multi-token; pure weight side)."""
    if model not in _EMB:
        from safetensors import safe_open
        p = embed_shard_path(model)
        with safe_open(p, framework='pt', device='cpu') as f:
            E = f.get_tensor('model.embed_tokens.weight').float().numpy()
        _EMB[model] = np.ascontiguousarray(E)
        log('  %s embed loaded %s (shard sha8 %s)' % (model, E.shape, sha8_file(p)))
    E = _EMB[model]
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(os.path.join(ROOT, 'models', 'hf', HF_DIR[model]))
    rows = []
    ntok_info = {}
    for w in words:
        ids = tok.encode(w, add_special_tokens=False)
        ntok_info[w] = len(ids)
        assert 1 <= len(ids) <= 8, ('token count out of band', model, w, ids)
        rows.append(E[np.array(ids, int)].astype(np.float32).mean(0))
    return np.stack(rows), ntok_info


def run_arm(sealed_3169, sealed_3189, sealed_3190):
    out = {}
    for m in MODELS_RUN:
        log('=== %s ===' % m)
        rd = int(sealed_3169['per_model'][m]['readout'])
        z = np.load(SRC[NPZ69_KEY[m]])
        H16 = z['H']
        NTz, NPz, NHz, Dz = H16.shape
        kout = NHz - 2
        assert kout == rd, ('readout slot', m)
        D = Dz
        Y = H16[:, :, kout, :].reshape(NT * NP_, D).astype(np.float32)
        assert np.isfinite(Y).all()

        tr_base = {}
        te_seen_all = {}
        for s in SEEDS_S1:
            tr_pairs, te_pairs = split_s1_246(s)
            tr_base[s] = [t * NP_ + PI_OF_PAIR[p] for t in range(NT)
                          for p in PAIRS if p in tr_pairs]
            te_seen_all[s] = [t * NP_ + PI_OF_PAIR[p] for t in range(NT)
                              for p in PAIRS if p in te_pairs]
        oov_rows = ALL_OOV_ROWS
        oov_row_cls = np.array([PAIRS[r % NP_][1] for r in oov_rows])

        def rv(r, ne):
            i, c = PAIRS[r % NP_]
            return phi_row(i, c, r // NP_, ne, NC, NT)

        E69 = sealed_3169['per_model'][m]['B']
        E89 = sealed_3189['per_model'][m]
        drift69 = 0.0
        Q_rows = np.zeros((len(oov_rows), D), dtype=np.float32)
        for s in SEEDS_S1:
            tr_rows = tr_base[s]
            Xtr = np.stack([rv(r, NE) for r in tr_rows])
            W = ridge_primal(Xtr, Y[tr_rows])
            ref = Y[tr_rows].mean(0)
            Dk = float(((Y[tr_rows] - ref) ** 2).sum(1).mean()) + 1e-9
            Xs = np.stack([rv(r, NE) for r in te_seen_all[s]])
            es = ((Xs @ W - Y[te_seen_all[s]]) ** 2).sum(1) / Dk
            Xo = np.stack([rv(r, NE) for r in oov_rows])
            eo = ((Xo @ W - Y[oov_rows]) ** 2).sum(1) / Dk
            drift69 = max(drift69, abs(float(es.mean()) - E69['E_seen_per_seed'][SEEDS_S1.index(s)]),
                          abs(float(eo.mean()) - E69['E_oov_per_seed'][SEEDS_S1.index(s)]))
            R_oov = Y[oov_rows] - Xo @ W
            for j in range(len(oov_rows)):
                rvv = R_oov[j]
                Q_rows[j] += rvv / (float(np.linalg.norm(rvv)) + 1e-18)
            del W, Xtr, Xs, Xo
        assert drift69 < REPL_TOL, ('3169 per-seed E replication', m, drift69)
        Qn = Q_rows / len(SEEDS_S1)
        Qn = Qn / (np.linalg.norm(Qn, axis=1, keepdims=True) + 1e-18)

        # 3189 scalar anchors
        s0 = np.linalg.svd(Qn, compute_uv=False)
        energy = s0 ** 2
        tot = float(energy.sum())
        top1 = float(energy[0] / tot)
        top2 = float(energy[:2].sum() / tot)
        cum = np.cumsum(energy) / tot
        r80 = int(np.searchsorted(cum, 0.8) + 1)
        r90 = int(np.searchsorted(cum, 0.9) + 1)
        cls_ids = list(range(NCLS_SEEN, NC))
        V = []
        within = []
        for c in cls_ids:
            idxs = np.where(oov_row_cls == c)[0]
            V.append(unit_mean(Qn[idxs]))
            Cc = Qn[idxs] @ Qn[idxs].T
            within.append(float(Cc[np.triu_indices(len(idxs), 1)].mean()))
        V = np.stack(V)
        Ccross = V @ V.T
        cross_med = float(np.median([float(Ccross[i, j])
                                     for i in range(4) for j in range(i + 1, 4)]))
        drift89 = max(abs(top1 - E89['top1']), abs(top2 - E89['top2']),
                      abs(cross_med - E89['cross_med']),
                      max(abs(a - b) for a, b in zip(within, E89['within_class'])),
                      abs(r80 - E89['r80']), abs(r90 - E89['r90']))
        assert drift89 == 0.0, ('3189 scalar drift', m, drift89)

        U, S, Vt = np.linalg.svd(Qn, full_matrices=False)
        ustar = Vt[0]

        # 3190 sealed anchors (loo + cos_vc + ks + loo_max_delta)
        E90 = sealed_3190['per_model'][m]
        loo = []
        for ci, c in enumerate(cls_ids):
            keep = oov_row_cls != c
            sl = np.linalg.svd(Qn[keep], compute_uv=False)
            el = sl ** 2
            loo.append(float(el[0] / el.sum()))
        loo_max = float(max(abs(x - top1) for x in loo))
        cos_vc = [float(ustar @ V[ci]) for ci in range(4)]
        drift90 = max(max(abs(a - b) for a, b in zip(loo, E90['loo_top1'])),
                      max(abs(a - b) for a, b in zip(cos_vc, E90['cos_ustar_vc'])),
                      abs(loo_max - E90['loo_max_delta']))
        assert drift90 == 0.0, ('3190 anchor drift', m, drift90)
        log('  device ok: drift69=%.2e drift89=0 drift90=0 (bitwise)' % drift69)

        # ---- NEW lexical arm: 10 class first-token embedding diff directions
        Erows, ntok_info = class_embed_rows(m, CLASSES)
        ntok_single_4b = None
        if m == 'qwen3-4b':
            assert all(v == 1 for v in ntok_info.values()), ('4b class words must be single-token', ntok_info)
            ntok_single_4b = True
        center = Erows.mean(0)
        G = np.stack([unit(Erows[c] - center) for c in range(NC)])  # 10xD
        cos_lex = [float(ustar @ G[c]) for c in range(NC)]
        abs_lex = [abs(x) for x in cos_lex]
        oov4_sum = float(sum(abs_lex[NCLS_SEEN:]))
        seen6_sum = float(sum(abs_lex[:NCLS_SEEN]))
        n_neg = int(sum(1 for x in cos_lex if x < 0))

        # control (a): random unit directions
        rng = np.random.RandomState(42)
        R = rng.standard_normal((N_RAND, D)).astype(np.float32)
        R = R / (np.linalg.norm(R, axis=1, keepdims=True) + 1e-18)
        rand_abs = np.abs(R @ ustar)
        rand_med = float(np.median(rand_abs))

        # control (b): pseudo-class word permutation (SMOKE-stage correction:
        # class-ROW permutation is degenerate - mean-centered diff directions
        # are invariant under row permutation, null == observed max; the
        # pseudo-class control shuffles the 73 ENTITY words into 10 groups)
        ENT_E, _ = class_embed_rows(m, ENTS)          # 73 x D entity word rows
        sizes = [8, 8, 8, 7, 7, 7, 7, 7, 7, 7]        # 3x8 + 7x7 = 73
        assert sum(sizes) == NE
        perm_max = []
        for it in range(N_PERM):
            pr = rng.permutation(NE)
            pseudo = np.stack([ENT_E[pr[pos:pos + gsz]].mean(0)
                               for gsz, pos in zip(sizes, np.cumsum([0] + sizes[:-1]))])
            Gp = np.stack([unit(pseudo[c] - pseudo.mean(0)) for c in range(NC)])
            perm_max.append(float(np.max(np.abs(Gp @ ustar))))
        perm_max = float(np.median(perm_max))

        gate_m = bool(max(abs_lex) >= GATE_RAND_MULT * rand_med and oov4_sum > seen6_sum)
        out[m] = dict(drift_3169=drift69, drift_3189=drift89, drift_3190=drift90,
                      top1=top1, cos_lex=cos_lex, abs_lex=abs_lex,
                      oov4_sum=oov4_sum, seen6_sum=seen6_sum, n_neg=n_neg,
                      rand_med=rand_med, perm_max_med=perm_max,
                      ntok=ntok_info, gate_model=gate_m)
        log('  cos_lex signed: %s' % ['%.4f' % x for x in cos_lex])
        log('  oov4_sum=%.4f seen6_sum=%.4f | max|cos|=%.4f vs rand_med=%.4f perm_med=%.4f | n_neg=%d | gate=%s'
            % (oov4_sum, seen6_sum, max(abs_lex), rand_med, perm_max, n_neg, gate_m))
    return out


def main():
    os.makedirs(OUTDIR, exist_ok=True)
    dsha = freeze()
    log('SMOKE=%s models=%s' % (SMOKE, MODELS_RUN))

    sealed_3169 = json.load(open(SRC['p3169_result'], encoding='utf-8'))
    sealed_3189 = json.load(open(SRC['p3189_result'], encoding='utf-8'))
    sealed_3190 = json.load(open(SRC['p3190_result'], encoding='utf-8'))
    for k, v in SHA_ANCHOR.items():
        actual = sha8_file(SRC[k])
        assert actual == v, ('src sha anchor', k, actual, v)
    log('src sha anchors OK')

    per_model = run_arm(sealed_3169, sealed_3189, sealed_3190)

    gate_all = bool(per_model and all(per_model[m]['gate_model'] for m in MODELS))
    verdict = 'ustar_lexical_registered' if (gate_all and not SMOKE) else \
              ('SMOKE_OK' if SMOKE else 'ustar_lexical_partial')
    log('GATE: %s verdict=%s' % (gate_all, verdict))

    res = {'phase': 3251, 'name': NAME, 'mode': 'smoke' if SMOKE else 'full',
           'design_sha8': dsha, 'per_model': per_model, 'gate_all': gate_all,
           'gate': {'rand_mult': GATE_RAND_MULT, 'models_needed': len(MODELS)},
           'verdict': verdict, 'zero_gpu': True,
           'src': 'p3251',
           'note': 'formalization of the 3190 downgraded lexical arm; '
                   'registry v1.33 byte-untouched (characterization/report-only)'}
    blob = json.dumps(res, ensure_ascii=False, sort_keys=True, indent=1).encode('utf-8')
    rsha = sha8(blob)
    res['res_sha8'] = rsha
    suffix = '_smoke' if SMOKE else ''
    json.dump(res, open(os.path.join(OUTDIR, 'result%s.json' % suffix), 'w', encoding='utf-8'),
              ensure_ascii=False, sort_keys=True, indent=1)
    if not SMOKE:
        np.savez_compressed(os.path.join(OUTDIR, 'ustar_lex.npz'),
                            allow_pickle=False,
                            **{('%s_cos_lex' % m): np.array(per_model[m]['cos_lex'], np.float64)
                               for m in MODELS_RUN})
    log('VERDICT: %s res_sha8=%s' % (verdict, rsha))
    open(os.path.join(OUTDIR, 'run_log%s.txt' % suffix), 'w', encoding='utf-8').write(
        '\n'.join(LOG) + '\n')


def unit_mean(rows_dn):
    v = rows_dn.mean(0)
    return v / (float(np.linalg.norm(v)) + 1e-18)


if __name__ == '__main__':
    main()
