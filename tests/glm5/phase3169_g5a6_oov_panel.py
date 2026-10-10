# -*- coding: utf-8 -*-
"""Phase 3169: G5-A6 out-of-vocabulary class panel (executes GAP-4 prereg).

Q03 protocol verbatim (B4 ridge one-hot additive predictor, lambda=1e-3,
normalized MSE, S1 split mechanism, batch=1 bf16 collect, fp16 H), panel
extended with 4 out-of-vocab classes (each 8 new entities) added to the 6
seen classes (41 entities verbatim from 3152).

Prereg correction (logged honestly): the prereg text listed '水果/金属' as
out-of-vocab classes, but both are IN the seen 3152 panel (claim_precision
error). OOV-ness is defined here by double exclusion: concept not in the 3152
six classes AND not in the 2881 joint-word-coordinate class families
(fruit/animal/metal/vehicle/country/food/nature/furniture/tool/clothing).
OOV classes frozen: 乐器/天气/运动/电器 (first tokens distinct from seen
classes; 门 unchanged: 2x / 1.5-2x borderline).

Gates (prereg, on 口径 B = class-level leave-out, the true-OOV reading):
  ratio = pooled(E_oov_B) / pooled(E_seen_B)   (pooled = mean over 3 models)
  ratio < 1.5          -> generalization holds (GAP-4 can close)
  ratio > 2.0          -> collapse confirmed (GAP-4 stays open, ratio quantified)
  1.5 <= ratio <= 2.0  -> borderline (re-judge with seeds 11/12)
口径 A (combo held-out on all 730 pairs, Q03-verbatim split mechanism) is an
appendix reading.

Device gates (SMOKE and formal):
  D0 vocab assertions (double exclusion)
  D1 collect finite, shapes exact
  D2 first tokens of 10 classes pairwise distinct
  D3 seen-row H bitwise equal to sealed 3152/3151 collect.npz (fp16)
  D4 anchor check: seen-subpanel (246 pairs, cols=51) Q03-verbatim recompute
     equals q03_result.json per-seed anchors (tol 1e-9)   [formal only]
SMOKE (P3169_SMOKE=1): qwen3-4b only, truncated panel (2 entities per class),
D0-D3 + pipeline sanity (D4 needs the full 41-entity subpanel).
"""
import os
import sys
import io
import json
import time
import hashlib
import datetime

import numpy as np

T0 = time.time()
try:
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
except Exception:
    pass

ROOT = r'D:\AI2050\Ai2050-OpenOne'
RDIR = os.path.join(ROOT, 'tests', 'glm5', 'result', 'rdc_query_construction_20260913')
B3152 = os.path.join(RDIR, 'phase3152', 'g1p2_tri_model_k1')
B3151 = os.path.join(RDIR, 'phase3151', 'g1p1_combo_additive_vs_interaction')
OUTDIR = os.path.join(RDIR, 'phase3169', 'g5a6_oov_panel')
SMOKE = os.environ.get('P3169_SMOKE', '') == '1'
LOG = []


def log(s):
    ln = '[%7.1f] %s' % (time.time() - T0, s)
    LOG.append(ln)
    try:
        print(ln, flush=True)
    except Exception:
        pass


def sha8(p):
    return hashlib.sha256(open(p, 'rb').read()).hexdigest()[:8]


# ---------------- panel (3152 verbatim seen + frozen OOV) ----------------
CLASSES_SEEN = ['水果', '动物', '交通工具', '家具', '金属', '颜色']
ENT_SEEN = {
    '水果': ['苹果', '香蕉', '梨', '西瓜', '葡萄', '草莓', '芒果', '柠檬'],
    '动物': ['狗', '猫', '老虎', '大象', '兔子', '猴子', '马', '牛'],
    '交通工具': ['汽车', '火车', '飞机', '摩托车', '卡车', '地铁'],
    '家具': ['桌子', '椅子', '床', '沙发', '地毯', '窗帘'],
    '金属': ['铁', '铜', '铝', '金', '银', '锌', '铅'],
    '颜色': ['红', '蓝', '绿', '黄', '黑', '白'],
}
TPL = {0: '{e}是一种{c}。',
       1: '{e}属于{c}这一类。',
       2: '{e}，一种常见的{c}。'}
SEEDS_S1 = [7, 8, 9]
SEEDS_BORDERLINE = [11, 12]
FRAC_S1 = 0.2
# 2881 class families (concept-level, from joint_word_coords target_list):
FAM2881 = ['fruit', 'animal', 'metal', 'vehicle', 'country', 'food',
           'nature', 'furniture', 'tool', 'clothing']
# frozen OOV classes (double exclusion; 电器 replaces prereg's 家电 because
# its first token 家 collides with 家具)
CLASSES_OOV = ['乐器', '天气', '运动', '电器']
ENT_OOV_FULL = {
    '乐器': ['钢琴', '小提琴', '吉他', '鼓', '笛子', '二胡', '琵琶', '口琴'],
    '天气': ['雨', '雪', '雷', '雾', '冰雹', '台风', '露水', '霜'],
    '运动': ['足球', '篮球', '乒乓球', '游泳', '跑步', '体操', '拳击', '围棋'],
    '电器': ['电视', '冰箱', '洗衣机', '空调', '微波炉', '电饭煲', '吸尘器', '风扇'],
}
ENT_OOV = {c: list(v) for c, v in ENT_OOV_FULL.items()}
assert len(ENT_SEEN['水果']) == 8 and sum(len(v) for v in ENT_SEEN.values()) == 41
assert sum(len(v) for v in ENT_OOV.values()) == 32

if SMOKE:
    CUT = 2
    ENT_SEEN = {c: v[:CUT] for c, v in ENT_SEEN.items()}
    ENT_OOV = {c: v[:CUT] for c, v in ENT_OOV.items()}

# full seen table (pre-truncation) for 3152 npz index mapping
ENT_SEEN_FULL = {cl: list(v) for cl, v in ENT_SEEN.items()} if not SMOKE else {
    '水果': ['苹果', '香蕉', '梨', '西瓜', '葡萄', '草莓', '芒果', '柠檬'],
    '动物': ['狗', '猫', '老虎', '大象', '兔子', '猴子', '马', '牛'],
    '交通工具': ['汽车', '火车', '飞机', '摩托车', '卡车', '地铁'],
    '家具': ['桌子', '椅子', '床', '沙发', '地毯', '窗帘'],
    '金属': ['铁', '铜', '铝', '金', '银', '锌', '铅'],
    '颜色': ['红', '蓝', '绿', '黄', '黑', '白'],
}
REF_ENT_INDEX = {}
_i = 0
for _cl in CLASSES_SEEN:
    for _e in ENT_SEEN_FULL[_cl]:
        REF_ENT_INDEX[_e] = _i
        _i += 1
assert _i == 41 and len(REF_ENT_INDEX) == 41

CLASSES = CLASSES_SEEN + CLASSES_OOV
ENT = dict(ENT_SEEN)
ENT.update(ENT_OOV)
ENTS = [e for cl in CLASSES for e in ENT[cl]]
CLS_OF = [CLASSES.index(cl) for cl in CLASSES for e in ENT[cl]]
NE = len(ENTS)
NC = len(CLASSES)
NT = len(TPL)
PAIRS = [(i, c) for i in range(NE) for c in range(NC)]
NP_ = len(PAIRS)
PANEL = NT * NP_
N_SEEN_ENT = sum(len(v) for v in ENT_SEEN.values())
N_OOV_ENT = sum(len(v) for v in ENT_OOV.values())
N_OOV_CLS = len(CLASSES_OOV)
N_OOV_PAIRS = N_OOV_ENT * N_OOV_CLS
N_SEEN_PAIRS = N_SEEN_ENT * len(CLASSES_SEEN)
N_OOVCLS_PAIRS = NE * N_OOV_CLS
N_NEWENT_PAIRS = N_OOV_ENT * len(CLASSES_SEEN)
assert N_SEEN_PAIRS == 246 or SMOKE, N_SEEN_PAIRS
assert N_OOVCLS_PAIRS == 292 or SMOKE, N_OOVCLS_PAIRS
assert N_NEWENT_PAIRS == 192 or SMOKE, N_NEWENT_PAIRS
KEEP_E = list(range(NE))
ALLP = set(PAIRS)
SEEN_PAIR_SET = set((i, c) for i in range(N_SEEN_ENT) for c in range(len(CLASSES_SEEN)))
# class-axis OOV rows: any entity x an OOV class (c >= 6)
OOV_CLS_PAIRS = set((i, c) for i in range(NE) for c in range(len(CLASSES_SEEN), NC))
# new-entity rows under seen classes (i >= N_SEEN_ENT, c < 6)
NEWENT_PAIRS = set((i, c) for i in range(N_SEEN_ENT, NE) for c in range(len(CLASSES_SEEN)))
PI_OF_PAIR = {p: j for j, p in enumerate(PAIRS)}
# seen rows aligned by ENTITY NAME to the 3152 npz layout (position i in the
# joint panel != position i in 3152 when SMOKE truncates per-class entities):
# joint pi = i_joint*NC+c, 3152 npz pi = REF_ENT_INDEX[entity]*6+c
JOINT_SEEN_PIS = []
REF_SEEN_PIS = []
for _i in range(N_SEEN_ENT):
    _e = ENTS[_i]
    _ir = REF_ENT_INDEX[_e]
    for _c in range(len(CLASSES_SEEN)):
        JOINT_SEEN_PIS.append(_i * NC + _c)
        REF_SEEN_PIS.append(_ir * len(CLASSES_SEEN) + _c)
assert len(JOINT_SEEN_PIS) == len(REF_SEEN_PIS) == N_SEEN_PAIRS

MODELS_ALL = [
    dict(name='qwen3-4b', mdir=os.path.join(ROOT, 'models', 'hf', 'qwen3-4b'),
         npz=os.path.join(B3152, 'qwen3-4b', 'collect.npz'), npz_sha8='4ef190b7',
         res=os.path.join(B3152, 'qwen3-4b', 'result.json')),
    dict(name='qwen3-14b', mdir=os.path.join(ROOT, 'models', 'hf', 'Qwen3-14B'),
         npz=os.path.join(B3152, 'qwen3-14b', 'collect.npz'), npz_sha8='733182c1',
         res=os.path.join(B3152, 'qwen3-14b', 'result.json')),
    dict(name='glm4-9b', mdir=os.path.join(ROOT, 'models', 'hf', 'glm4-9b-chat-hf'),
         npz=os.path.join(B3151, 'collect.npz'), npz_sha8='c711946c',
         res=os.path.join(B3152, 'glm4k1', 'result.json')),
]
MODELS = MODELS_ALL[:1] if SMOKE else MODELS_ALL

Q03 = json.load(io.open(os.path.join(ROOT, 'tests', 'deepseek', 'result', 'q03_result.json'),
                        encoding='utf-8'))

# full-panel constants (DESIGN must be identical for SMOKE and formal runs)
NE_FULL = 41 + 32
NC_FULL = 10
PAIRS_FULL = NE_FULL * NC_FULL
PANEL_ROWS_FULL = 3 * PAIRS_FULL
N_OOVCLS_PAIRS_FULL = NE_FULL * 4
N_NEWENT_PAIRS_FULL = 32 * 6

DESIGN = dict(
    phase=3169, name='g5a6_oov_panel', zero_gpu=False,
    prereg_source='GAP-4 prereg in gap_ledger_v1.json (3168, f92ed0cd) / MEMO 3168',
    prereg_correction='水果/金属 listed in prereg text are seen classes; OOV defined by '
                      'double exclusion (3152 six classes + 2881 class families %s); '
                      '门 unchanged (2x / 1.5-2x borderline)' % FAM2881,
    classes_seen=CLASSES_SEEN, ent_seen_full=ENT_SEEN_FULL,
    classes_oov=CLASSES_OOV, ent_oov_full=ENT_OOV_FULL,
    tpl=TPL, ne_full=NE_FULL, nc_full=NC_FULL, nt=NT, pairs_full=PAIRS_FULL,
    panel_rows_full=PANEL_ROWS_FULL,
    protocol='Q03 verbatim: B4 ridge one-hot[entity]+one-hot[class]+one-hot[template]+bias, '
             'lam=1e-3, fp32 solve, Dk=train-row variance mean, E=normalized MSE at readout '
             'layer, batch=1 bf16 collect, H fp16, last-token hidden states',
    readout_layer='per-model from 3152/3151 k1_model_report.readout',
    splits=dict(
        B_class_leaveout='train = seen 246 pairs minus S1 test (seed s); test_seen = S1 test; '
                         'test_oov = all 292 class-axis-OOV rows (c>=6, entity x OOV class, '
                         'fully absent from train); test_newent = 192 new-entity rows under '
                         'seen classes; X cols = NE+NC+NT+1 (joint feature space)',
        A_combo_heldout='S1 split mechanism on all 730 pairs (seed s, frac 0.2); E_seen_A / '
                        'E_oov_A by class membership of test pairs',
        anchor_check='seen subpanel 246 pairs, cols=51 verbatim Q03 phi_main, seeds 7/8/9, '
                     'kout; must equal q03 per-seed anchors tol 1e-9',
    ),
    gate=dict(
        metric='ratio_B = pooled(E_oov_B)/pooled(E_seen_B), pooled=mean over models',
        bands='<1.5 generalization (GAP-4 closable); >2.0 collapse confirmed (GAP-4 open); '
              '[1.5,2.0] borderline -> re-judge with seeds 11/12',
    ),
    device_gates=dict(D0='vocab double exclusion', D1='collect finite+shapes', D2='first tokens '
                      'distinct', D3='seen-row H bitwise vs 3152/3151 npz', D4='anchor check '
                      '1e-9 (formal only)'),
    seeds_s1=SEEDS_S1, seeds_borderline=SEEDS_BORDERLINE, frac_s1=FRAC_S1,
    models_full=[dict(name=m['name'], mdir=m['mdir'].replace(ROOT, ''), npz_sha8=m['npz_sha8'])
                 for m in MODELS_ALL],
)


def freeze():
    os.makedirs(OUTDIR, exist_ok=True)
    exep = os.path.join(OUTDIR, 'execution.json')
    body = {k: v for k, v in DESIGN.items()}
    raw = json.dumps(body, ensure_ascii=False, indent=1, sort_keys=True)
    d8 = hashlib.sha256(raw.encode('utf-8')).hexdigest()[:8]
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


# ---------------- Q03 verbatim mechanics (parameterized) ----------------
def split_s1_246(seed):
    """Q03 verbatim: permutation on the seen pairs, frac 0.2 -> test set."""
    allp = SEEN_PAIR_SET
    pairs246 = sorted(allp)
    assert len(pairs246) == N_SEEN_PAIRS, 'seen pairs count'
    rng = np.random.RandomState(seed)
    idx = rng.permutation(len(pairs246))
    n_test = int(round(FRAC_S1 * len(pairs246)))
    test = set(pairs246[j] for j in idx[:n_test])
    return allp - test, test


def split_s1_joint(seed):
    rng = np.random.RandomState(seed)
    idx = rng.permutation(NP_)
    n_test = int(round(FRAC_S1 * NP_))
    test = set(PAIRS[j] for j in idx[:n_test])
    return ALLP - test, test


def rows_of_joint(pair_set):
    return [t * NP_ + PI_OF_PAIR[p] for t in range(NT) for p in PAIRS if p in pair_set]


def ridge_primal(Xtr, Ytr, lam=1e-3):
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


# ---------------- main ----------------
def main():
    d8 = freeze()
    log('== Phase 3169 g5a6_oov_panel %s ==' % ('SMOKE' if SMOKE else 'FULL'))
    log('panel: %d ents x %d classes x %d tpl = %d rows (seen %d ents/%d pairs, oov %d ents/%d pairs)'
        % (NE, NC, NT, PANEL, N_SEEN_ENT, N_SEEN_PAIRS, N_OOV_ENT, N_OOV_PAIRS))

    # ---- D0 vocab double exclusion ----
    seen_words = set(ENTS[:N_SEEN_ENT])
    oov_words = set(ENTS[N_SEEN_ENT:])
    assert not (seen_words & oov_words), 'seen/oov word overlap'
    # OOV class concept not among seen classes (string level)
    for c in CLASSES_OOV:
        assert c not in CLASSES_SEEN, ('oov class collides seen', c)
    # OOV entity not among seen entities
    for w in oov_words:
        assert w not in seen_words, ('oov entity collides seen', w)
    # concept-level exclusion vs 2881 families is logged as the frozen mapping
    log('D0 vocab OK: %d oov words, classes %s; 2881 exclusion frozen by DESIGN (concept map)'
        % (len(oov_words), CLASSES_OOV))

    per_model = {}
    for m in MODELS:
        name = m['name']
        log('--- model %s ---' % name)
        car_ok = sha8(m['npz']) == m['npz_sha8']
        assert car_ok, ('carrier sha mismatch', name)
        rp = json.load(io.open(m['res'], encoding='utf-8-sig'))['k1_model_report']
        kout = int(rp['readout'])
        kstar = int(rp['kstar'])
        anchor_ps = [float(x) for x in rp['b4_rel_readout_per_seed']]
        anchor_mean = float(rp['b4_rel_readout_mean3seed'])

        import torch
        from transformers import AutoTokenizer, AutoModelForCausalLM
        torch.manual_seed(0)
        cfg = json.load(io.open(os.path.join(m['mdir'], 'config.json'), encoding='utf-8'))
        NL = cfg['num_hidden_layers']
        D = cfg['hidden_size']
        NH = NL + 1
        assert kout < NH and kstar < NH

        tok = AutoTokenizer.from_pretrained(m['mdir'], trust_remote_code=True)

        def ids_of(text):
            return tok(text, add_special_tokens=False)['input_ids']

        CLS_TOK = [ids_of(c)[0] for c in CLASSES]
        assert len(set(CLS_TOK)) == NC, ('D2 first-token collision', CLS_TOK)
        log('D2 first tokens OK (%s)' % CLS_TOK)
        prompts = [TPL[t].format(e=ENTS[i], c=CLASSES[c])
                   for t in range(NT) for (i, c) in PAIRS]
        assert len(prompts) == PANEL
        TOKIDS = [ids_of(p) for p in prompts]

        cache = os.path.join(OUTDIR, 'collect_smoke_%s.npz' % name if SMOKE
                             else 'collect_%s.npz' % name)
        if os.path.exists(cache):
            z = np.load(cache)
            H16 = z['H']
            MARG = z['marg']
            log('collect cache hit %s H=%s' % (os.path.basename(cache), H16.shape))
        else:
            model = AutoModelForCausalLM.from_pretrained(
                m['mdir'], dtype=torch.bfloat16, trust_remote_code=True).to('cuda').eval()
            assert model.config.num_hidden_layers == NL
            H16 = np.zeros((NT, NP_, NH, D), np.float16)
            MARG = np.zeros((NT, NP_, NC), np.float32)
            with torch.no_grad():
                for t in range(NT):
                    for pi in range(NP_):
                        ii = torch.tensor([TOKIDS[t * NP_ + pi]], device='cuda')
                        o = model(input_ids=ii, output_hidden_states=True)
                        hs = o.hidden_states
                        hv = np.stack([h[0, -1].float().detach().cpu().numpy()
                                       for h in hs], 0)
                        H16[t, pi] = hv.astype(np.float16)
                        lg = o.logits[0, -1].float().detach().cpu().numpy()
                        cl = lg[[CLS_TOK[k] for k in range(NC)]]
                        MARG[t, pi] = cl - cl.mean()
                        del o, hs, hv, lg, cl
                    log('collect tpl%d done' % t)
            np.savez_compressed(cache, H=H16, marg=MARG)
            del model
            torch.cuda.empty_cache()
            log('collect saved %s (%d B)' % (os.path.basename(cache), os.path.getsize(cache)))

        # ---- D1 finite + shapes ----
        assert H16.shape == (NT, NP_, NH, D), ('D1 shape', H16.shape)
        assert np.isfinite(H16.astype(np.float32)).all(), 'D1 non-finite H'
        log('D1 finite OK H=%s' % (H16.shape,))

        # ---- D3 seen-row H bitwise vs sealed npz ----
        zref = np.load(m['npz'])
        Href = zref['H']
        assert Href.shape[0] == NT and Href.shape[2] == NH and Href.shape[3] == D, \
            ('ref H shape', Href.shape)
        seen_pi_set = set(JOINT_SEEN_PIS)
        n_bit = 0
        n_bad = 0
        for t in range(NT):
            for pj, jref in zip(JOINT_SEEN_PIS, REF_SEEN_PIS):
                a = H16[t, pj]
                b = Href[t, jref]
                n_bit += 1
                if not np.array_equal(a, b):
                    n_bad += 1
        chk3 = (n_bad == 0)
        log('D3 seen-row H bitwise: %d rows compared, %d mismatch -> %s'
            % (n_bit, n_bad, 'OK' if chk3 else 'FAIL'))
        assert chk3, ('D3 H bitwise mismatch', n_bad)

        # ---- D4 anchor check (formal only): cols=51 verbatim on seen subpanel ----
        anchor_ok = None
        rec_anchor = None
        drift = None
        if not SMOKE:
            def Y_sub(k):
                # (NT*246, D) in q03 row order: t*246 + pi3152
                Ys = np.zeros((NT * len(REF_SEEN_PIS), D), np.float32)
                for t in range(NT):
                    for j, pj in enumerate(REF_SEEN_PIS):
                        Ys[t * len(REF_SEEN_PIS) + j] = H16[t, JOINT_SEEN_PIS[j], k]
                return Ys

            ne_s, nc_s = N_SEEN_ENT, len(CLASSES_SEEN)
            pairs_s = [(i, c) for i in range(ne_s) for c in range(nc_s)]
            nps = len(pairs_s)
            assert nps == 246
            rec_anchor = []
            for s in SEEDS_S1:
                # q03 verbatim split on the 246 pairs
                rng = np.random.RandomState(s)
                idx = rng.permutation(nps)
                n_test = int(round(FRAC_S1 * nps))
                te_pairs = set(pairs_s[j] for j in idx[:n_test])
                tr_pairs = set(pairs_s) - te_pairs
                tr_rows = [t * nps + j for t in range(NT)
                           for j, p in enumerate(pairs_s) if p in tr_pairs]
                te_rows = [t * nps + j for t in range(NT)
                           for j, p in enumerate(pairs_s) if p in te_pairs]
                cols = ne_s + nc_s + NT + 1

                def rv(t, pi):
                    i, c = pairs_s[pi]
                    return phi_row(i, c, t, ne_s, nc_s, NT)

                Xtr = np.stack([rv(r // nps, r % nps) for r in tr_rows])
                Y = Y_sub(kout)
                W = ridge_primal(Xtr, Y[tr_rows], lam=1e-3)
                Xte = np.stack([rv(r // nps, r % nps) for r in te_rows])
                B4te = Xte @ W
                ref = Y[tr_rows].mean(0)
                Dk = float(((Y[tr_rows] - ref) ** 2).sum(1).mean()) + 1e-9
                e = ((B4te - Y[te_rows]) ** 2).sum(1) / Dk
                rec_anchor.append(float(e.mean()))
            drift = [abs(a - b) for a, b in zip(anchor_ps, rec_anchor)]
            anchor_ok = max(drift) < 1e-9 and abs(float(np.mean(rec_anchor)) - anchor_mean) < 1e-9
            log('D4 anchor check: recompute=%s anchor=%s max_drift=%.3e -> %s'
                % (['%.9f' % x for x in rec_anchor], ['%.9f' % x for x in anchor_ps],
                   max(drift), 'OK' if anchor_ok else 'FAIL'))
            assert anchor_ok, ('D4 anchor drift', max(drift))

        # ---- E_read 口径 B (class leave-out) ----
        def Y_at(k):
            return H16[:, :, k, :].reshape(NT * NP_, D).astype(np.float32)

        Y = Y_at(kout)
        cols_B = phi_cols(NE, NC, NT)

        def rv_B(r):
            t, pi = r // NP_, r % NP_
            i, c = PAIRS[pi]
            return phi_row(i, c, t, NE, NC, NT)

        eB = dict(E_seen_per_seed=[], E_oov_per_seed=[], E_newent_per_seed=[])
        for s in SEEDS_S1:
            tr_pairs, te_pairs = split_s1_246(s)
            tr_rows = [t * NP_ + PI_OF_PAIR[p] for t in range(NT) for p in PAIRS if p in tr_pairs]
            te_seen_rows = [t * NP_ + PI_OF_PAIR[p] for t in range(NT) for p in PAIRS
                            if p in te_pairs]
            te_oov_rows = rows_of_joint(OOV_CLS_PAIRS)
            te_newent_rows = rows_of_joint(NEWENT_PAIRS)
            assert len(tr_pairs) + len(te_pairs) == len(SEEN_PAIR_SET)
            assert not (set(tr_pairs) & OOV_CLS_PAIRS) and not (set(te_pairs) & OOV_CLS_PAIRS), \
                'B: oov-class rows leaked into train/test_seen'
            Xtr = np.stack([rv_B(r) for r in tr_rows])
            W = ridge_primal(Xtr, Y[tr_rows], lam=1e-3)
            ref = Y[tr_rows].mean(0)
            Dk = float(((Y[tr_rows] - ref) ** 2).sum(1).mean()) + 1e-9
            Xs = np.stack([rv_B(r) for r in te_seen_rows])
            Xo = np.stack([rv_B(r) for r in te_oov_rows])
            Xn = np.stack([rv_B(r) for r in te_newent_rows])
            es = ((Xs @ W - Y[te_seen_rows]) ** 2).sum(1) / Dk
            eo = ((Xo @ W - Y[te_oov_rows]) ** 2).sum(1) / Dk
            en = ((Xn @ W - Y[te_newent_rows]) ** 2).sum(1) / Dk
            eB['E_seen_per_seed'].append(float(es.mean()))
            eB['E_oov_per_seed'].append(float(eo.mean()))
            eB['E_newent_per_seed'].append(float(en.mean()))
        E_seen_B = float(np.mean(eB['E_seen_per_seed']))
        E_oov_B = float(np.mean(eB['E_oov_per_seed']))
        E_newent_B = float(np.mean(eB['E_newent_per_seed']))

        # ---- E_read 口径 A (combo held-out on all pairs) ----
        eA = dict(E_seen_per_seed=[], E_oov_per_seed=[])
        for s in SEEDS_S1:
            tr_pairs, te_pairs = split_s1_joint(s)
            tr_rows = [t * NP_ + PI_OF_PAIR[p] for t in range(NT) for p in PAIRS if p in tr_pairs]
            te_rows = [t * NP_ + PI_OF_PAIR[p] for t in range(NT) for p in PAIRS if p in te_pairs]
            te_seen = [r for r in te_rows if PAIRS[r % NP_] in SEEN_PAIR_SET]
            te_oov = [r for r in te_rows if PAIRS[r % NP_] in OOV_CLS_PAIRS]
            Xtr = np.stack([rv_B(r) for r in tr_rows])
            W = ridge_primal(Xtr, Y[tr_rows], lam=1e-3)
            ref = Y[tr_rows].mean(0)
            Dk = float(((Y[tr_rows] - ref) ** 2).sum(1).mean()) + 1e-9
            Xs = np.stack([rv_B(r) for r in te_seen])
            Xo = np.stack([rv_B(r) for r in te_oov])
            es = ((Xs @ W - Y[te_seen]) ** 2).sum(1) / Dk
            eo = ((Xo @ W - Y[te_oov]) ** 2).sum(1) / Dk
            eA['E_seen_per_seed'].append(float(es.mean()))
            eA['E_oov_per_seed'].append(float(eo.mean()))
        E_seen_A = float(np.mean(eA['E_seen_per_seed']))
        E_oov_A = float(np.mean(eA['E_oov_per_seed']))
        log('%s B: E_seen=%.6f E_oov=%.6f E_newent=%.6f ratio=%.4f | A: E_seen=%.6f E_oov=%.6f ratio=%.4f'
            % (name, E_seen_B, E_oov_B, E_newent_B, E_oov_B / E_seen_B, E_seen_A, E_oov_A,
               E_oov_A / E_seen_A))

        per_model[name] = dict(
            readout=kout, kstar=kstar, H_shape=list(H16.shape),
            carrier_sha8=m['npz_sha8'], d3_bitwise_rows=n_bit, d3_mismatch=n_bad,
            anchor=dict(per_seed_recompute=rec_anchor, per_seed_anchor=anchor_ps,
                        drift=drift, ok=bool(anchor_ok) if rec_anchor else None,
                        mean3seed_anchor=anchor_mean),
            B=dict(E_seen_per_seed=eB['E_seen_per_seed'], E_oov_per_seed=eB['E_oov_per_seed'],
                   E_newent_per_seed=eB['E_newent_per_seed'],
                   E_seen=E_seen_B, E_oov=E_oov_B, E_newent=E_newent_B,
                   ratio=E_oov_B / E_seen_B),
            A=dict(E_seen_per_seed=eA['E_seen_per_seed'], E_oov_per_seed=eA['E_oov_per_seed'],
                   E_seen=E_seen_A, E_oov=E_oov_A, ratio=E_oov_A / E_seen_A),
        )
        del H16, MARG, Y, zref, Href
        import gc
        gc.collect()

    # ---------------- gate ----------------
    ratios_B = [per_model[k]['B']['ratio'] for k in per_model]
    Eo = [per_model[k]['B']['E_oov'] for k in per_model]
    Es = [per_model[k]['B']['E_seen'] for k in per_model]
    pooled_oov = float(np.mean(Eo))
    pooled_seen = float(np.mean(Es))
    ratio = pooled_oov / pooled_seen
    if SMOKE:
        gate_verdict = 'smoke_pipeline_only'
    elif ratio < 1.5:
        gate_verdict = 'generalization_holds_gap4_closable'
    elif ratio > 2.0:
        gate_verdict = 'collapse_confirmed_gap4_open'
    else:
        gate_verdict = 'borderline_rejudge_seeds_11_12'
    log('GATE: pooled E_oov_B=%.6f pooled E_seen_B=%.6f ratio=%.4f -> %s'
        % (pooled_oov, pooled_seen, ratio, gate_verdict))

    verdict = 'g5a6_oov_panel|%s|%d_models|ratio_B=%.4f|%s' % (
        'smoke' if SMOKE else 'full', len(per_model), ratio, gate_verdict)

    summary = dict(
        phase=3169, name='g5a6_oov_panel', smoke=SMOKE, design_sha8=d8,
        verdict=verdict,
        panel=dict(ne=NE, nc=NC, nt=NT, pairs=NP_, panel_rows=PANEL,
                   classes_oov=CLASSES_OOV, n_oov_pairs=N_OOV_PAIRS),
        gate=dict(pooled_E_oov_B=pooled_oov, pooled_E_seen_B=pooled_seen,
                  ratio_B=ratio, verdict=gate_verdict),
        per_model=per_model,
        prereg_correction=DESIGN['prereg_correction'],
    )
    raw = json.dumps(summary, ensure_ascii=False, indent=1, sort_keys=True)
    res8 = hashlib.sha256(raw.encode('utf-8')).hexdigest()[:8]
    mid = json.dumps(dict(summary, res_sha8=res8), ensure_ascii=False, indent=1, sort_keys=True)
    seal8 = hashlib.sha256(mid.encode('utf-8')).hexdigest()[:8]
    rpath = os.path.join(OUTDIR, 'smoke_result.json' if SMOKE else 'result.json')
    with io.open(rpath, 'w', encoding='utf-8') as f:
        f.write(json.dumps(dict(summary, res_sha8=res8, seal_sha8=seal8),
                           ensure_ascii=False, indent=1, sort_keys=True))
    log('result written res=%s seal=%s' % (res8, seal8))
    log('verdict: %s' % verdict)
    with io.open(os.path.join(OUTDIR, 'smoke_run_log.txt' if SMOKE else 'run_log.txt'),
                 'w', encoding='utf-8') as f:
        f.write('\n'.join(LOG) + '\n')


if __name__ == '__main__':
    main()
