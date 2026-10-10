# -*- coding: utf-8 -*-
"""Phase 3175: G5-A12 structural residual localization - cross arms.

Pre-registered (MEMO Phase 3174 tail + prereg draft 8abb8d6d, Phase 3174 G8):
3172 verdict borderline_partial_recovery left a structural residual of about
1.8x after port calibration (k=8 pooled ratio 1.8002, port_removed_frac
29.09%). Two candidate explanations:
  H1 entity_familiarity: 3172 calibration entities came from the frozen panel
     vocabulary - possibly less familiar to the models than panel test
     entities, so recovery was capped; out-panel calibration entities would
     recover even less (ratio_out > ratio_in, delta > 0.15).
  H2 port_residual: the one-hot class port mechanism can only be partially
     learned regardless of entity source; both arms show the same recovery
     shape (|delta| <= 0.15).

Arms (both follow Phase 3172 protocol verbatim: Q03 ridge, B-protocol, seeds
7/8/9, RandomState(seed+1000) reset per class, no replacement):
  in  arm: calibration entities sampled from the 3169 panel vocabulary
     (8/class); reuses the sealed 3169 collect npz; ZERO new forward. The
     sealed 3172 pooled curve must be replayed bitwise (drift < 1e-9).
  out arm: calibration entities sampled from a NEW out-panel vocabulary
     (10/class, frozen in DESIGN before any GPU forward); H rows collected
     with the 3169 verbatim chain (batch=1 bf16, full-layer last-token, fp16
     H). phi entity columns are extended dynamically by the sampled out
     entities (ne = 73 + 4k); k=0 has zero extension and must reproduce the
     sealed 3169 gate bitwise (drift < 1e-9).

Test-set parity: te_oov = ALL_OOV_ROWS - in_arm_calibration_rows for BOTH
arms (the in-arm calibration rows are in the train set, so they leave the
test pool in both arms; out calibration rows live outside the panel so
nothing further is removed). E_seen (S1 test) and E_newent (panel OOV
entities under seen classes) use identical row sets in both arms. Hence
E_oov is directly comparable across arms.

Primary gate (pre-registered):
  delta = pooled_ratio_out(k=8) - pooled_ratio_in(k=8)
  |delta| <= 0.15 -> port_residual_dominant   (H2; mechanism note final)
  delta  >  0.15  -> entity_familiarity_component_confirmed (H1)
  delta  < -0.15  -> anomaly_register (device review before any claim)

Immutable predicates:
  k=0 both arms reproduce 3169 gate ratio_B = 2.5388114997805062 (1e-9);
  out vocabulary frozen at freeze time, before any GPU forward;
  in arm replays the sealed 3172 curve bitwise.
DESIGN is fully static (3169 discipline).
"""
import hashlib
import io
import json
import os
import sys
import time

import numpy as np

T0 = time.time()
PHASE = 3175
NAME = 'g5a12_residual_arms'
SMOKE = os.environ.get('P3175_SMOKE', '') == '1'
ROOT = r'D:\AI2050\Ai2050-OpenOne'
RDIR = os.path.join(ROOT, 'tests', 'glm5', 'result', 'rdc_query_construction_20260913')
P3169 = os.path.join(RDIR, 'phase3169', 'g5a6_oov_panel')
P3172 = os.path.join(RDIR, 'phase3172', 'g5a9_port_calibration')
P3174 = os.path.join(RDIR, 'phase3174', 'g5a11_audit')
OUTDIR = os.path.join(RDIR, 'phase3175', NAME)
LOG = []

try:
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
except Exception:
    pass


def log(s):
    ln = '[%7.1f] %s' % (time.time() - T0, s)
    LOG.append(ln)
    try:
        print(ln, flush=True)
    except Exception:
        pass


def sha8(p):
    return hashlib.sha256(open(p, 'rb').read()).hexdigest()[:8]


MODELS = [
    dict(name='qwen3-4b', mdir=os.path.join(ROOT, 'models', 'hf', 'qwen3-4b'),
         npz=os.path.join(P3169, 'collect_qwen3-4b.npz')),
    dict(name='qwen3-14b', mdir=os.path.join(ROOT, 'models', 'hf', 'Qwen3-14B'),
         npz=os.path.join(P3169, 'collect_qwen3-14b.npz')),
    dict(name='glm4-9b', mdir=os.path.join(ROOT, 'models', 'hf', 'glm4-9b-chat-hf'),
         npz=os.path.join(P3169, 'collect_glm4-9b.npz')),
]
SMOKE_NPZ = os.path.join(P3169, 'collect_smoke_qwen3-4b.npz')

SHA_ANCHOR = {
    'p3169_result': '5b51c2c1', 'p3169_smoke_result': '466ef808',
    'p3169_4b': '36eb4ff0', 'p3169_14b': '97f98575', 'p3169_glm4': '9cc3f8b4',
    'p3169_smoke_npz': 'db50c9dc',
    'p3172_result': 'd8ddc481', 'p3172_smoke_result': 'f7fa4445',
    'p3172_exec': 'b5684d0b',
    'p3174_prereg3175': '8abb8d6d', 'p3174_result': '131a4d61',
}

# ---------------- panel (3169/3172 verbatim) ---------------------------------
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
CLASSES_OOV = ['乐器', '天气', '运动', '电器']
ENT_OOV_PANEL = {
    '乐器': ['钢琴', '小提琴', '吉他', '鼓', '笛子', '二胡', '琵琶', '口琴'],
    '天气': ['雨', '雪', '雷', '雾', '冰雹', '台风', '露水', '霜'],
    '运动': ['足球', '篮球', '乒乓球', '游泳', '跑步', '体操', '拳击', '围棋'],
    '电器': ['电视', '冰箱', '洗衣机', '空调', '微波炉', '电饭煲', '吸尘器', '风扇'],
}
# NEW out-panel vocabulary: frozen HERE, before any GPU forward (prereg
# predicate). Same double-exclusion criteria as 3169 (classes already passed;
# entities are new members of the same OOV classes, disjoint from every panel
# word and from each other; probe p3175 verified string-level sanity).
ENT_OOV_OUT = {
    '乐器': ['长笛', '竖琴', '唢呐', '大提琴', '手风琴', '萨克斯', '木琴', '锣', '钹', '竖笛'],
    '天气': ['彩虹', '闪电', '暴雨', '微风', '寒潮', '热浪', '沙尘暴', '霜冻', '梅雨', '阴天'],
    '运动': ['网球', '排球', '跳水', '滑雪', '射箭', '击剑', '马拉松', '瑜伽', '跳高', '举重'],
    '电器': ['烤箱', '豆浆机', '加湿器', '电吹风', '热水器', '打印机', '电熨斗', '榨汁机', '路由器', '电磁炉'],
}
NT = 3
SEEDS_S1 = [7, 8, 9]
FRAC_S1 = 0.2
LAM = 1e-3
# full-vocabulary copies for DESIGN (must be run-mode independent: the DESIGN
# hash must be identical for SMOKE and formal runs - 3169 discipline)
ENT_OUT_FULL = {c: list(v) for c, v in ENT_OOV_OUT.items()}
assert sum(len(v) for v in ENT_OUT_FULL.values()) == 40
KS_FULL = [0, 1, 2, 4, 8]
KS_SMOKE = [0, 1, 2]
KS = KS_SMOKE if SMOKE else KS_FULL
if SMOKE:
    ENT_SEEN = {c: v[:2] for c, v in ENT_SEEN.items()}
    ENT_OOV_PANEL = {c: v[:2] for c, v in ENT_OOV_PANEL.items()}
    ENT_OOV_OUT = {c: v[:2] for c, v in ENT_OOV_OUT.items()}
CLASSES = CLASSES_SEEN + CLASSES_OOV
ENT = dict(ENT_SEEN)
ENT.update(ENT_OOV_PANEL)
ENTS = [e for cl in CLASSES for e in ENT[cl]]
NE = len(ENTS)
NC = len(CLASSES)
PAIRS = [(i, c) for i in range(NE) for c in range(NC)]
NP_ = len(PAIRS)
N_SEEN_ENT = sum(len(v) for v in ENT_SEEN.values())
N_SEEN_PAIRS = N_SEEN_ENT * len(CLASSES_SEEN)
if not SMOKE:
    assert (NE, NC, NP_) == (73, 10, 730), 'panel'
    assert N_SEEN_PAIRS == 246, 'seen pairs'
SEEN_PAIR_SET = set((i, c) for i in range(N_SEEN_ENT) for c in range(len(CLASSES_SEEN)))
PI_OF_PAIR = {p: j for j, p in enumerate(PAIRS)}
ALL_OOV_ROWS = [t * NP_ + pi for t in range(NT) for pi in range(NP_)
                if PAIRS[pi][1] >= len(CLASSES_SEEN)]
# out panel
OUT_ENTS = [e for cl in CLASSES_OOV for e in ENT_OOV_OUT[cl]]
N_OUT_E = len(OUT_ENTS)
OUT_PAIRS = [(i, c) for i in range(N_OUT_E) for c in range(NC)]
NP_OUT = len(OUT_PAIRS)
PANEL_ROW_BASE = {}
_i = 0
for cl in CLASSES:
    for e in ENT[cl]:
        PANEL_ROW_BASE[e] = _i
        _i += 1
assert _i == NE
N_OOV_PANEL_PER_CLS = {cl: len(ENT_OOV_PANEL[cl]) for cl in CLASSES_OOV}
N_OOV_OUT_PER_CLS = {cl: len(ENT_OOV_OUT[cl]) for cl in CLASSES_OOV}

# ---------------- DESIGN (fully static) ---------------------------------------
DESIGN = dict(
    phase=PHASE, name=NAME, zero_gpu=False,
    prereg_source='MEMO Phase 3174 tail (缺口排序裁决) + prereg_3175_structural'
                  '_residual_draft.json (Phase 3174 G8, file sha8 8abb8d6d)',
    hypotheses=dict(
        H1_entity_familiarity='panel-internal calibration entities recover '
                              'deeper than out-panel ones (ratio_in < '
                              'ratio_out, delta > 0.15)',
        H2_port_residual='both arms share the same recovery shape '
                         '(|delta| <= 0.15): the residual is port-intrinsic'),
    arms=dict(
        in_panel='calibration entities sampled from the 3169 panel OOV '
                 'vocabulary (8 per class); sealed 3169 collect npz reused, '
                 'zero new forward; sealed 3172 pooled curve replayed bitwise',
        out_panel='calibration entities sampled from the NEW out-panel '
                  'vocabulary (10 per class, frozen below before any GPU '
                  'forward); H rows collected with the 3169 verbatim chain '
                  '(batch=1 bf16, full-layer last-token hidden states, fp16 '
                  'H); phi entity columns extended dynamically by the '
                  'sampled out entities (ne = 73 + 4k), k=0 zero extension',
        test_set_parity='te_oov = ALL_OOV_ROWS minus in-arm calibration rows '
                        'for BOTH arms; E_seen and E_newent row sets '
                        'identical across arms; E_oov directly comparable',
    ),
    ent_out_full=ENT_OUT_FULL,
    n_out_per_cls_full=10,
    out_panel_size_full=40,
    intervention=dict(
        unit='per OOV class c: k entities sampled with RandomState(seed+1000) '
             'reset per class, no replacement; in-arm samples the 8-entity '
             'panel pool (3172 verbatim), out-arm samples the 10-entity '
             'out-panel pool; their rows (all NT templates) join the '
             'B-protocol train set; k=0 = no calibration (3169 replication)',
        k_grid_full=[0, 1, 2, 4, 8], k_grid_smoke=[0, 1, 2]),
    protocol='Q03 verbatim: phi_row one-hot[entity]+one-hot[class]+one-hot'
             '[template]+bias, ridge_primal lam=1e-3 fp32, Dk = train-row '
             'variance mean + 1e-9, E = normalized MSE at the readout slot; '
             'seeds 7/8/9; ratio = E_oov/E_seen recomputed per (arm,k) under '
             'the same W/Dk chain; pooled = mean over 3 models',
    gate_pre_registered=dict(
        primary='delta = pooled_ratio_out(k=8) - pooled_ratio_in(k=8): '
                '|delta| <= 0.15 -> port_residual_dominant (H2; GAP-4 '
                'mechanism_note finalized); delta > 0.15 -> '
                'entity_familiarity_component_confirmed (H1); delta < -0.15 '
                '-> anomaly_register (device review before any claim)',
        secondary='arm k-gradient shapes (saturation), E_newent monitor, '
                  'per-class consistency (4 classes same direction)',
        device='k=0 both arms reproduce sealed 3169 gate pooled values '
               '(drift < 1e-9); in arm replays sealed 3172 pooled curve '
               'bitwise; out collect self-check (3-prompt bitwise reself)',
    ),
    immutable_predicates=[
        'out vocabulary frozen in DESIGN before any GPU forward',
        'k=0 arms reproduce 3169 gate ratio_B = 2.5388114997805062 within '
        '1e-9', 'in arm replays 3172 sealed pooled curve within 1e-9',
    ],
    slots=dict(kout='per-model from sealed 3169 result per_model.readout'),
    dtype_chain='H float16 -> float32 (Y) -> fp32 ridge chain (3169 verbatim)',
    source_sha8=SHA_ANCHOR,
    smoke='SMOKE (P3175_SMOKE=1): qwen3-4b only, truncated panel and out '
          'pools (2 per class), k in [0,1,2]; in-arm k=0 replays the sealed '
          '3172 smoke pooled values; out smoke collect 240 forwards',
)


def freeze():
    os.makedirs(OUTDIR, exist_ok=True)
    exep = os.path.join(OUTDIR, 'execution.json')
    core = {k: v for k, v in DESIGN.items() if k not in ('created', 'design_sha8')}
    d8 = hashlib.sha256(json.dumps(core, ensure_ascii=False, indent=1,
                                   sort_keys=True).encode('utf-8')).hexdigest()[:8]
    if os.path.exists(exep):
        prev = json.load(io.open(exep, encoding='utf-8'))
        if prev.get('design_sha8') != d8:
            raise SystemExit('DRIFT: execution.json design_sha8 %s != current %s'
                             % (prev.get('design_sha8'), d8))
        log('freeze: existing execution.json OK (%s)' % d8)
    else:
        body = dict(core)
        body['created'] = time.strftime('%Y-%m-%d %H:%M:%S')
        body['design_sha8'] = d8
        with io.open(exep, 'w', encoding='utf-8', newline='\n') as f:
            f.write(json.dumps(body, ensure_ascii=False, indent=1, sort_keys=True))
        log('freeze: execution.json written design_sha8=%s' % d8)
    return d8


# ---------------- Q03 verbatim mechanics (3172 copy) --------------------------
def split_s1_246(seed):
    allp = SEEN_PAIR_SET
    pairs246 = sorted(allp)
    assert len(pairs246) == N_SEEN_PAIRS, 'seen pairs count'
    rng = np.random.RandomState(seed)
    idx = rng.permutation(len(pairs246))
    n_test = int(round(FRAC_S1 * len(pairs246)))
    test = set(pairs246[j] for j in idx[:n_test])
    return allp - test, test


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


def calib_panel(seed, k):
    """3172 verbatim: per OOV class k panel entity global indices."""
    out = {}
    for ci, cl in enumerate(CLASSES_OOV):
        n_c = len(ENT[cl])
        base = PANEL_ROW_BASE[ENT[cl][0]]
        if k == 0:
            out[ci] = []
        else:
            assert k <= n_c, ('k exceeds class entity count', k, cl)
            rng = np.random.RandomState(seed + 1000)
            pick = rng.choice(n_c, size=k, replace=False)
            out[ci] = sorted(base + int(j) for j in pick)
    return out


def calib_out(seed, k):
    """Same protocol, out-panel pool (10 per class) -> out-pool indices."""
    out = {}
    for ci, cl in enumerate(CLASSES_OOV):
        n_c = len(ENT_OOV_OUT[cl])
        if k == 0:
            out[ci] = []
        else:
            assert k <= n_c, ('k exceeds out pool size', k, cl)
            rng = np.random.RandomState(seed + 1000)
            pick = rng.choice(n_c, size=k, replace=False)
            out[ci] = sorted(int(j) for j in pick)
    return out


def run():
    d8 = freeze()
    # ---- G_anchor ----
    src = {
        'p3169_result': os.path.join(P3169, 'result.json'),
        'p3169_smoke_result': os.path.join(P3169, 'smoke_result.json'),
        'p3169_4b': MODELS[0]['npz'], 'p3169_14b': MODELS[1]['npz'],
        'p3169_glm4': MODELS[2]['npz'],
        'p3169_smoke_npz': SMOKE_NPZ,
        'p3172_result': os.path.join(P3172, 'result.json'),
        'p3172_smoke_result': os.path.join(P3172, 'smoke_result.json'),
        'p3172_exec': os.path.join(P3172, 'execution.json'),
        'p3174_prereg3175': os.path.join(P3174, 'prereg_3175_structural_residual_draft.json'),
        'p3174_result': os.path.join(P3174, 'result.json'),
    }
    for kk, v in SHA_ANCHOR.items():
        got = sha8(src[kk])
        assert got == v, ('G_anchor', kk, got, v)
    log('G_anchor: %d files OK' % len(SHA_ANCHOR))
    R69 = json.load(io.open(src['p3169_result'], encoding='utf-8'))
    S69 = json.load(io.open(src['p3169_smoke_result'], encoding='utf-8'))
    R72 = json.load(io.open(src['p3172_result'], encoding='utf-8'))
    S72 = json.load(io.open(src['p3172_smoke_result'], encoding='utf-8'))
    R74 = json.load(io.open(src['p3174_result'], encoding='utf-8'))
    assert R69['res_sha8'] == '49430a39' and S69['res_sha8'] == '46600d73'
    assert R72['res_sha8'] == '463f42c8' and R72['seal_sha8'] == 'cdb85525'
    assert S72['res_sha8'] == '493ea55f'
    assert R74['res_sha8'] == '199c3f5b' and R74['seal_sha8'] == 'b08bfeee'
    assert R72['design_sha8'] == '3912db14'
    REF72 = S72 if SMOKE else R72
    REF69 = S69 if SMOKE else R69
    log('content sha OK; REF72 verdict: %s' % REF72['verdict'])

    per_model = {}
    for m in (MODELS[:1] if SMOKE else MODELS):
        mk = m['name']
        log('=== model %s ===' % mk)
        import torch
        from transformers import AutoTokenizer, AutoModelForCausalLM
        torch.manual_seed(0)
        cfg = json.load(io.open(os.path.join(m['mdir'], 'config.json'), encoding='utf-8'))
        NL = cfg['num_hidden_layers']
        D = cfg['hidden_size']
        NH = NL + 1
        rd = int(REF69['per_model'][mk]['readout'])
        kout = NL - 1
        assert kout == rd, ('G_slot', kout, rd)
        log('G_slot: kout=%d == 3169 readout; NL=%d D=%d' % (kout, NL, D))

        tok = AutoTokenizer.from_pretrained(m['mdir'], trust_remote_code=True)

        def ids_of(text):
            return tok(text, add_special_tokens=False)['input_ids']

        CLS_TOK = [ids_of(c)[0] for c in CLASSES]
        assert len(set(CLS_TOK)) == NC, ('D2 first-token collision', CLS_TOK)
        log('D2 first tokens OK (%s)' % CLS_TOK)

        # ---- panel H (in arm, sealed npz) ----
        z = np.load(SMOKE_NPZ if SMOKE else m['npz'])
        H16 = z['H']
        NTz, NPz, NHz, Dz = H16.shape
        assert NTz == NT and NPz == NP_ and NHz == NH and Dz == D, ('shape', H16.shape)
        assert np.isfinite(H16.astype(np.float32)).all(), 'H non-finite'
        Y = H16[:, :, kout, :].reshape(NT * NP_, D).astype(np.float32)
        log('panel H loaded %s' % (H16.shape,))

        # ---- out collect (3169 verbatim chain) ----
        cache = os.path.join(OUTDIR, 'collect_out_smoke_%s.npz' % mk if SMOKE
                             else 'collect_out_%s.npz' % mk)
        prompts_out = [TPL[t].format(e=OUT_ENTS[i], c=CLASSES[c])
                       for t in range(NT) for (i, c) in OUT_PAIRS]
        assert len(prompts_out) == NT * NP_OUT
        TOKIDS_OUT = [ids_of(p) for p in prompts_out]
        fresh = False
        if os.path.exists(cache):
            zo = np.load(cache)
            H_out = zo['H']
            MARG_out = zo['marg']
            log('out collect cache hit %s H=%s' % (os.path.basename(cache), H_out.shape))
        else:
            model = AutoModelForCausalLM.from_pretrained(
                m['mdir'], dtype=torch.bfloat16, trust_remote_code=True).to('cuda').eval()
            assert model.config.num_hidden_layers == NL
            H_out = np.zeros((NT, NP_OUT, NH, D), np.float16)
            MARG_out = np.zeros((NT, NP_OUT, NC), np.float32)
            with torch.no_grad():
                for t in range(NT):
                    for pj in range(NP_OUT):
                        ii = torch.tensor([TOKIDS_OUT[t * NP_OUT + pj]], device='cuda')
                        o = model(input_ids=ii, output_hidden_states=True)
                        hs = o.hidden_states
                        hv = np.stack([h[0, -1].float().detach().cpu().numpy()
                                       for h in hs], 0)
                        H_out[t, pj] = hv.astype(np.float16)
                        lg = o.logits[0, -1].float().detach().cpu().numpy()
                        cl = lg[[CLS_TOK[k] for k in range(NC)]]
                        MARG_out[t, pj] = cl - cl.mean()
                        del o, hs, hv, lg, cl
                    log('out collect tpl%d done' % t)
            np.savez_compressed(cache, H=H_out, marg=MARG_out)
            # D_out2: 3-prompt bitwise reself (deterministic bf16 batch=1)
            for (t, pj) in ((0, 0), (0, 1), (1, 0)):
                ii = torch.tensor([TOKIDS_OUT[t * NP_OUT + pj]], device='cuda')
                o = model(input_ids=ii, output_hidden_states=True)
                hv2 = np.stack([h[0, -1].float().detach().cpu().numpy()
                                for h in o.hidden_states], 0)
                assert np.array_equal(hv2.astype(np.float16), H_out[t, pj]), \
                    ('D_out2 reself mismatch', mk, t, pj)
                del o
            log('D_out2 reself bitwise OK (3 prompts)')
            del model
            torch.cuda.empty_cache()
            log('out collect saved %s (%d B)' % (os.path.basename(cache), os.path.getsize(cache)))
            fresh = True
        assert H_out.shape == (NT, NP_OUT, NH, D), ('D_out1 shape', H_out.shape)
        assert np.isfinite(H_out.astype(np.float32)).all(), 'D_out1 non-finite'
        assert np.isfinite(MARG_out).all(), 'D_out1 non-finite MARG'
        log('D_out1 finite OK H_out=%s fresh=%s' % (H_out.shape, fresh))
        Y_out = H_out[:, :, kout, :].reshape(NT * NP_OUT, D).astype(np.float32)

        te_seen_all = []
        tr_base = {}
        for s in SEEDS_S1:
            tr_pairs, te_pairs = split_s1_246(s)
            tr_rows = [t * NP_ + PI_OF_PAIR[p] for t in range(NT)
                       for p in PAIRS if p in tr_pairs]
            te_rows = [t * NP_ + PI_OF_PAIR[p] for t in range(NT)
                       for p in PAIRS if p in te_pairs]
            tr_base[s] = tr_rows
            te_seen_all.append(te_rows)

        def rv(r, ne):
            i, c = PAIRS[r % NP_]
            return phi_row(i, c, r // NP_, ne, NC, NT)

        kres = {'in': {}, 'out': {}}
        for k in KS:
            arms = {}
            for arm in ('in', 'out'):
                per_seed = dict(E_seen=[], E_oov=[], E_newent=[],
                                E_oov_cls=[[] for _ in range(len(CLASSES_OOV))])
                for si, s in enumerate(SEEDS_S1):
                    in_cal = calib_panel(s, k)
                    in_cal_rows = []
                    for ci, idxs in in_cal.items():
                        c = len(CLASSES_SEEN) + ci
                        for i in idxs:
                            for t in range(NT):
                                in_cal_rows.append(t * NP_ + PI_OF_PAIR[(i, c)])
                    assert len(set(in_cal_rows)) == len(in_cal_rows)
                    assert not (set(in_cal_rows) & set(tr_base[s]))
                    # test-set parity: both arms drop the in-arm cal rows
                    te_oov = [r for r in ALL_OOV_ROWS if r not in set(in_cal_rows)]
                    if arm == 'in':
                        tr_rows_k = tr_base[s] + in_cal_rows
                        ne_ext = NE
                        X_list = [rv(r, ne_ext) for r in tr_rows_k]
                        Y_list = [Y[r] for r in tr_rows_k]
                    else:
                        out_cal = calib_out(s, k)
                        ext_ents = []
                        out_triples = []
                        for ci in range(len(CLASSES_OOV)):
                            c = len(CLASSES_SEEN) + ci
                            for j in out_cal[ci]:
                                ext_ents.append((c, j))
                                for t in range(NT):
                                    out_triples.append((j, c, t))
                        # ext_ents sorted by (class, pool index); classes are
                        # disjoint pools so the union has exactly 4k members
                        assert len(ext_ents) == len(CLASSES_OOV) * k or k == 0
                        ext_ents = sorted(ext_ents)
                        ext_index = {}
                        for rank, (c, j) in enumerate(ext_ents):
                            ext_index[(c, j)] = NE + rank
                        ne_ext = NE + len(ext_ents)
                        X_list = [rv(r, ne_ext) for r in tr_base[s]]
                        Y_list = [Y[r] for r in tr_base[s]]
                        for (j, c, t) in out_triples:
                            g = ext_index[(c, j)]
                            v = np.zeros(phi_cols(ne_ext, NC, NT), np.float32)
                            v[g] = 1.0
                            v[ne_ext + c] = 1.0
                            v[ne_ext + NC + t] = 1.0
                            v[-1] = 1.0
                            X_list.append(v)
                            Y_list.append(Y_out[t * NP_OUT + j * NC + c])
                    Xtr = np.stack(X_list)
                    Ytr = np.stack(Y_list)
                    W = ridge_primal(Xtr, Ytr)
                    ref = Ytr.mean(0)
                    Dk = float(((Ytr - ref) ** 2).sum(1).mean()) + 1e-9
                    Xs = np.stack([rv(r, ne_ext) for r in te_seen_all[si]])
                    es = ((Xs @ W - Y[te_seen_all[si]]) ** 2).sum(1) / Dk
                    Xo = np.stack([rv(r, ne_ext) for r in te_oov])
                    eo = ((Xo @ W - Y[te_oov]) ** 2).sum(1) / Dk
                    ne_rows = [t * NP_ + pi for t in range(NT) for pi in range(NP_)
                               if PAIRS[pi][0] >= N_SEEN_ENT and PAIRS[pi][1] < 6]
                    Xn = np.stack([rv(r, ne_ext) for r in ne_rows])
                    en = ((Xn @ W - Y[ne_rows]) ** 2).sum(1) / Dk
                    per_seed['E_seen'].append(float(es.mean()))
                    per_seed['E_oov'].append(float(eo.mean()))
                    per_seed['E_newent'].append(float(en.mean()))
                    for ci in range(len(CLASSES_OOV)):
                        rows_c = [r for r in te_oov
                                  if PAIRS[r % NP_][1] == len(CLASSES_SEEN) + ci]
                        Xc = np.stack([rv(r, ne_ext) for r in rows_c])
                        ec = ((Xc @ W - Y[rows_c]) ** 2).sum(1) / Dk
                        per_seed['E_oov_cls'][ci].append(float(ec.mean()))
                E_seen = float(np.mean(per_seed['E_seen']))
                E_oov = float(np.mean(per_seed['E_oov']))
                E_newent = float(np.mean(per_seed['E_newent']))
                arms[arm] = dict(E_seen=E_seen, E_oov=E_oov, E_newent=E_newent,
                                 ratio=E_oov / E_seen,
                                 E_seen_per_seed=per_seed['E_seen'],
                                 E_oov_per_seed=per_seed['E_oov'],
                                 E_oov_cls_mean=[float(np.mean(v))
                                                 for v in per_seed['E_oov_cls']])
                log('k=%d arm=%s: E_seen=%.6f E_oov=%.6f ratio=%.4f E_newent=%.6f'
                    % (k, arm, E_seen, E_oov, E_oov / E_seen, E_newent))
                del Xtr, W
            kres['in'][k] = arms['in']
            kres['out'][k] = arms['out']

        # ---- G_k0 per model (in arm) ----
        refB = REF72['per_model'][mk]['kcurves'][str(0)]
        for key in ('E_oov', 'E_seen', 'E_newent'):
            d0 = abs(kres['in'][0][key] - float(refB[key]))
            assert d0 < 1e-9, ('G_k0 in', mk, key, d0)
            log('G_k0 in %s %s: drift=%.2e OK' % (mk, key, d0))
        # out arm k=0 must equal in arm k=0 exactly (zero-extension path)
        for key in ('E_oov', 'E_seen', 'E_newent'):
            d0 = abs(kres['out'][0][key] - kres['in'][0][key])
            assert d0 == 0.0, ('G_k0 out path parity', mk, key, d0)
        log('G_k0 out path parity: exact 0.0 (zero-extension degenerate OK)')
        # in-arm replay: full k grid vs sealed 3172
        for k in KS:
            refk = REF72['per_model'][mk]['kcurves'][str(k)]
            for key in ('E_oov', 'E_seen', 'E_newent', 'ratio'):
                dr = abs(kres['in'][k][key] - float(refk[key]))
                assert dr < 1e-9, ('G_in_replay', mk, k, key, dr)
        log('G_in_replay %s: k grid x4 keys all drift<1e-9 vs sealed 3172' % mk)

        per_model[mk] = dict(NL=NL, D=D, kout=kout,
                             in_curve=kres['in'], out_curve=kres['out'])
        del z, H16, Y, H_out, Y_out

    # ---- pooled ----
    pooled = {'in': {}, 'out': {}}
    for arm in ('in', 'out'):
        for k in KS:
            po = float(np.mean([per_model[mk][arm + '_curve'][k]['E_oov']
                                for mk in per_model]))
            ps = float(np.mean([per_model[mk][arm + '_curve'][k]['E_seen']
                                for mk in per_model]))
            pooled[arm][k] = dict(E_oov=po, E_seen=ps, ratio=po / ps)
            log('pooled %s k=%d: E_oov=%.6f E_seen=%.6f ratio=%.4f'
                % (arm, k, po, ps, po / ps))

    # ---- G_k0 pooled (both arms vs sealed 3169 gate / 3172 smoke) ----
    g69 = REF69['gate']
    for arm in ('in', 'out'):
        for key, pk in (('pooled_E_oov_B', 'E_oov'), ('pooled_E_seen_B', 'E_seen')):
            d0 = abs(pooled[arm][0][pk] - float(g69[key]))
            assert d0 < 1e-9, ('G_k0 pooled', arm, key, d0)
            log('G_k0 pooled %s %s: drift=%.2e OK' % (arm, key, d0))
        dr = abs(pooled[arm][0]['ratio'] - float(g69['ratio_B']))
        assert dr < 1e-9, ('G_k0 pooled ratio', arm, dr)
        log('G_k0 pooled %s ratio_B: drift=%.2e OK' % (arm, dr))

    # ---- in-arm pooled replay vs sealed 3172 ----
    for k in KS:
        refp = REF72['pooled'][str(k)]
        for key in ('E_oov', 'E_seen', 'ratio'):
            dr = abs(pooled['in'][k][key] - float(refp[key]))
            assert dr < 1e-9, ('G_in_replay pooled', k, key, dr)
    log('G_in_replay pooled: k grid x3 keys all drift<1e-9 vs sealed 3172')

    # ---- primary gate ----
    kmax = KS[-1]
    r_in = pooled['in'][kmax]['ratio']
    r_out = pooled['out'][kmax]['ratio']
    delta = r_out - r_in
    if SMOKE:
        main_cls = 'smoke'
    else:
        if abs(delta) <= 0.15:
            main_cls = 'port_residual_dominant'
        elif delta > 0.15:
            main_cls = 'entity_familiarity_component_confirmed'
        else:
            main_cls = 'anomaly_register'
    verdict = ('g5a12_residual_arms|%s|%d_models|delta_k%s=%.4f|'
               'ratio_in_k%s=%.4f|ratio_out_k%s=%.4f|%s'
               % ('smoke' if SMOKE else 'full', len(per_model), kmax, delta,
                  kmax, r_in, kmax, r_out, main_cls))
    log('VERDICT: %s' % verdict)

    # per-class direction consistency (secondary, formal only)
    percls = None
    if not SMOKE:
        percls = {}
        for ci, cl in enumerate(CLASSES_OOV):
            rec = []
            for mk in per_model:
                e_in = per_model[mk]['in_curve'][kmax]['E_oov_cls_mean'][ci]
                e_out = per_model[mk]['out_curve'][kmax]['E_oov_cls_mean'][ci]
                rec.append(e_out - e_in)
            percls[cl] = dict(delta_per_model=rec,
                              all_positive=bool(all(v > 0 for v in rec)),
                              all_negative=bool(all(v < 0 for v in rec)))
        log('per-class deltas: %s' % json.dumps(percls))

    result = dict(phase=PHASE, name=NAME, design_sha8=d8, smoke=SMOKE,
                  k_grid=KS, per_model=per_model, pooled=pooled,
                  delta_gate=dict(delta=delta, threshold=0.15, main_cls=main_cls,
                                  ratio_in_kmax=r_in, ratio_out_kmax=r_out,
                                  rule='|delta|<=0.15 port_residual_dominant / '
                                       '>0.15 entity_familiarity_component_'
                                       'confirmed / <-0.15 anomaly_register',
                                  direction='H1 predicts delta>0 (out entities '
                                            'less familiar, weaker recovery); '
                                            'H2 predicts |delta|<=0.15'),
                  per_class_delta=percls,
                  overall=dict(delta=delta, main_cls=main_cls),
                  verdict=verdict)
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
