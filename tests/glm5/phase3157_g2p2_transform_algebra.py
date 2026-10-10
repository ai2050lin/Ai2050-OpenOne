# -*- coding: utf-8 -*-
# Phase 3157 (G2-P2): 变换代数 P1 —— 算子对易子（T_N x T_R x T_C）
# 预注册: AGI_GPT5_MEMO 2026-10-08（3156 closeout 时冻结，观测前）:
#   (1) 装置合并: T_N=否定(句中极性), T_R=关系(is-a/has-a), T_C=上下文(k=16 真实前缀 vs 空);
#       T_P≈恒等已由 3156 B 臂确立(位置算子=单位元)
#   (2) 面板: 16 实体(3155 SHARED 前 16) x 2 关系(isa/hasa) x 2 极性(+/-) x 2 上下文(0/k16)
#       = 128 行/模型, 三模型(4b/14b/glm4); SMOKE 前 4 实体 = 32 行
#   (3) 度量(KOUT+全层):
#       对易子(关系差分) exch_R = mean_e ||dR_e(+)-dR_e(-)|| / mean_e mean(||dR_e(+)||,||dR_e(-)||),
#         dR_e(pol) = h(e,isa,pol)-h(e,hasa,pol);  C=0 与 C=1 分算
#       对易子(否定差分) exch_N 同构, dN_e(rel) = h(e,rel,-)-h(e,rel,+)
#       平衡 ANOVA 序贯块 [E16|R1|N1|C1|ExR16|ExN16|ExC16|RxN1|RxC1|NxC1|残差], RxN 单列=代数扭曲方差视角
#       T_C 关系无关性: dC_e,r,n = h(C=1)-h(C=0) 跨 (r,n) 两两 cos 均值 + 范数比
#       null: 实体 shuffle 100 次 -> exch null 分布
#   (4) 门(KOUT, C=0): max(exch_R, exch_N) < 0.5 -> commutative_algebra_supported;
#       0.5-2 -> partial; > 2 -> non_commutative(更强发现)
#   (5) summary: 三模型 KOUT ANOVA 份额向量(11 维)两两 Pearson >= 0.8 + exch 跨模型
# 模板(zh): isa '{E}是一种{C}。'/'{E}不是一种{C}。'; hasa '{E}有{P}。'/'{E}没有{P}。'
# 采集: last-token 全隐层 H fp16 (NROWS, NL+1, D); 确定性锚 3 行位级
# 教训内置: SMOKE 目录分离; design 全 str 键 JSON; fail-fast 断言; 正交块 orthonorm 链; numpy2 solve 尾维
import os, sys, json, time, hashlib
import numpy as np

T0 = time.time()
try:
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
except Exception:
    pass
MODEL = os.environ.get('P3157_MODEL') or (sys.argv[1] if len(sys.argv) > 1 else 'qwen3-4b')
SMOKE = os.environ.get('P3157_SMOKE') == '1'
ROOT = r'D:\AI2050\Ai2050-OpenOne'
RDIR = os.path.join(ROOT, 'tests', 'glm5', 'result', 'rdc_query_construction_20260913')
NAME = 'g2p2_transform_algebra_commutator'
BASE = os.path.join(RDIR, 'phase3157', NAME, MODEL)
if SMOKE:
    BASE = os.path.join(BASE, 'smoke')
os.makedirs(BASE, exist_ok=True)
LOGP = os.path.join(BASE, 'run_log.txt')

def log(s):
    ln = '[%7.1f] %s' % (time.time() - T0, s)
    with open(LOGP, 'a', encoding='utf-8') as f:
        f.write(ln + '\n')
    try:
        print(ln, flush=True)
    except Exception:
        pass

def freeze_design(phase_name, design):
    eblob = json.dumps(design, ensure_ascii=False, sort_keys=True, indent=1).encode('utf-8')
    sha = hashlib.sha256(eblob).hexdigest()
    exe_p = os.path.join(BASE, 'execution.json')
    if os.path.exists(exe_p):
        prev = json.load(open(exe_p, encoding='utf-8'))
        assert prev['design_sha'] == sha, 'DESIGN DRIFT'
        log('execution.json match (sha %s)' % sha[:8])
    else:
        json.dump({'phase': 3157, 'name': phase_name, 'design_sha': sha,
                   'design': design, 'frozen_before': 'any model observation',
                   'created': time.strftime('%Y-%m-%d %H:%M:%S')},
                  open(exe_p, 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
        log('execution.json FROZEN (sha %s)' % sha[:8])
    return sha

def seal_result(result, out_name):
    blob = json.dumps(result, ensure_ascii=False, indent=1, sort_keys=True).encode('utf-8')
    res_sha8 = hashlib.sha256(blob).hexdigest()[:8]
    result['res_sha8'] = res_sha8
    result['verdict'] = result['verdict'] + '|sha8_' + res_sha8
    rp = os.path.join(BASE, out_name)
    json.dump(result, open(rp, 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
    seal = hashlib.sha256(open(rp, 'rb').read()).hexdigest()[:8]
    result['seal_sha8'] = seal
    json.dump(result, open(rp, 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
    log('RESULT %s res_sha8=%s seal=%s verdict=%s' % (out_name, res_sha8, seal, result['verdict']))
    return res_sha8, seal

# ---------------- 冻结材料 ----------------
SHARED = [
    ('桌子', '家具', '桌腿'), ('椅子', '座椅', '椅背'), ('窗户', '建筑构件', '窗框'),
    ('门', '建筑构件', '门把手'), ('硬币', '货币', '图案'), ('轮胎', '橡胶配件', '花纹'),
    ('纸', '材料', '纤维'), ('书', '出版物', '书页'), ('刀', '工具', '刀刃'),
    ('碗', '餐具', '碗底'), ('衣服', '服饰', '袖子'), ('鞋', '日用品', '鞋底'),
    ('桥', '建筑', '桥墩'), ('房子', '建筑', '屋顶'), ('汽车', '交通工具', '车轮'),
    ('钟表', '计时工具', '表盘'), ('杯子', '容器', '杯柄'), ('瓶子', '容器', '瓶盖'),
    ('帽子', '服饰', '帽檐'), ('篮子', '容器', '提手'), ('灯笼', '照明用具', '骨架'),
    ('锁', '保安用具', '锁孔'), ('钥匙', '五金用品', '齿纹'), ('伞', '雨具', '伞骨'),
]
N_E = 4 if SMOKE else 16
ENTS = SHARED[:N_E]
PREFIX_ZH = '今天天气很好，我们在讨论语言模型如何表示语言。'
K_CTX = 16
TPL = {
    ('isa', '+'): '{E}是一种{C}。', ('isa', '-'): '{E}不是一种{C}。',
    ('hasa', '+'): '{E}有{P}。', ('hasa', '-'): '{E}没有{P}。',
}
EXCH_GATE_LO, EXCH_GATE_HI = 0.5, 2.0
FP_GATE = 0.8
N_SHUFFLE = 100

def text_of(rel, pol, ent):
    e, c, p = ent
    return TPL[(rel, pol)].replace('{E}', e).replace('{C}', c).replace('{P}', p)

NROWS = N_E * 2 * 2 * 2
RELS = ['isa', 'hasa']
POLs = ['+', '-']
CTXs = [0, 1]
MAT_NOTES = ['ents=%d rels=2 pols=2 ctx=2 rows=%d' % (N_E, NROWS),
             'negation adds 1-2 chars (declared, absorbed by N main effect)',
             'ctx = %d-token real prefix (3156 zh prefix cycled)' % K_CTX]

ROWS = []
for ei, ent in enumerate(ENTS):
    for ri, rel in enumerate(RELS):
        for pi, pol in enumerate(POLs):
            for ci, cx in enumerate(CTXs):
                ROWS.append(dict(ei=ei, rel=rel, pol=pol, ctx=cx,
                                 prompt=text_of(rel, pol, ent)))
assert len(ROWS) == NROWS
# 最小对断言: 同 (e,rel,ctx) 下 +/- 句都非空且不同
for ei in range(N_E):
    for rel in RELS:
        for cx in CTXs:
            a = text_of(rel, '+', ENTS[ei])
            b = text_of(rel, '-', ENTS[ei])
            assert a != b and len(a) >= 4 and len(b) >= 4, ('min pair', ei, rel, cx, a, b)

# ---------------- 模式 A/B/C: GPU 采集 + 分析 ----------------
if MODEL in ('qwen3-4b', 'qwen3-14b', 'glm4'):
    MDIR_MAP = {'qwen3-4b': 'qwen3-4b', 'qwen3-14b': 'Qwen3-14B', 'glm4': 'glm4-9b-chat-hf'}
    MDIR = os.path.join(ROOT, 'models', 'hf', MDIR_MAP[MODEL])
    cfg = json.load(open(os.path.join(MDIR, 'config.json'), encoding='utf-8'))
    NL = cfg['num_hidden_layers']
    HID = cfg['hidden_size']
    KOUT = NL - 1
    KSTAR = int(round(0.075 * NL))
    design = dict(model=MODEL, mdir=MDIR, phase=NAME, nl=NL, hidden=HID, readout=KOUT,
                  kstar=KSTAR, ents=[list(e) for e in ENTS], rels=RELS, pols=POLs,
                  ctx=[0, 1], k_ctx=K_CTX, prefix=PREFIX_ZH, tpl={'%s|%s' % k: v for k, v in TPL.items()},
                  n_rows=NROWS, exch_gate=[EXCH_GATE_LO, EXCH_GATE_HI], fp_gate=FP_GATE,
                  n_shuffle=N_SHUFFLE, material_asserts=MAT_NOTES, smoke=SMOKE,
                  pre_reg='MEMO 2026-10-08 (G2-P2 transform algebra commutator); frozen before observation')
    exe_sha = freeze_design('g2p2_%s' % MODEL, design)
    log('model=%s NL=%d D=%d readout=%d rows=%d' % (MODEL, NL, HID, KOUT, NROWS))

    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
    torch.manual_seed(0)
    tok = AutoTokenizer.from_pretrained(MDIR, trust_remote_code=True)
    log('model load begin')
    model = AutoModelForCausalLM.from_pretrained(
        MDIR, dtype=torch.bfloat16, trust_remote_code=True).to('cuda').eval()
    log('model loaded: %s' % type(model).__name__)
    assert model.config.num_hidden_layers == NL
    D = HID
    NH = NL + 1

    pre_ids = tok(PREFIX_ZH, add_special_tokens=False)['input_ids']
    ctx_ids = (pre_ids * (K_CTX // len(pre_ids) + 1))[:K_CTX]
    assert len(ctx_ids) == K_CTX

    cache = os.path.join(BASE, 'collect_smoke.npz' if SMOKE else 'collect.npz')
    det_note = 'cache_hit (determinism not rechecked this run)'
    if os.path.exists(cache):
        z = np.load(cache)
        H16 = z['H']
        log('collect cache hit %s H=%s' % (os.path.basename(cache), H16.shape))
    else:
        H16 = np.zeros((NROWS, NH, D), np.float16)
        with torch.no_grad():
            for i, r in enumerate(ROWS):
                ids = (ctx_ids if r['ctx'] == 1 else []) + \
                      tok(r['prompt'], add_special_tokens=False)['input_ids']
                ii = torch.tensor([ids], dtype=torch.int64, device='cuda')
                o = model(input_ids=ii, output_hidden_states=True)
                hs = o.hidden_states
                H16[i] = np.stack([h[0, -1].float().detach().cpu().numpy()
                                   for h in hs], 0).astype(np.float16)
                del o, hs
                if (i + 1) % 64 == 0:
                    log('collect %d/%d' % (i + 1, NROWS))
        det_rows = [0, NROWS // 2, NROWS - 1]
        det_ok = True
        det_max = 0.0
        with torch.no_grad():
            for i in det_rows:
                r = ROWS[i]
                ids = (ctx_ids if r['ctx'] == 1 else []) + \
                      tok(r['prompt'], add_special_tokens=False)['input_ids']
                ii = torch.tensor([ids], dtype=torch.int64, device='cuda')
                o = model(input_ids=ii, output_hidden_states=True)
                hv = np.stack([h[0, -1].float().detach().cpu().numpy()
                               for h in o.hidden_states], 0).astype(np.float16)
                dmax = float(np.abs(hv.astype(np.float32) - H16[i].astype(np.float32)).max())
                det_max = max(det_max, dmax)
                if dmax != 0.0:
                    det_ok = False
                del o
        det_note = 'bitwise' if det_ok else 'fp16 max abs diff %.3e (tolerance pass)' % det_max
        assert det_max < 1e-3, ('determinism check fail', det_max)
        log('determinism recheck rows=%s -> %s' % (det_rows, det_note))
        np.savez_compressed(cache, H=H16,
                            ei=np.array([r['ei'] for r in ROWS], np.int16),
                            rel=np.array([RELS.index(r['rel']) for r in ROWS], np.int8),
                            pol=np.array([POLs.index(r['pol']) for r in ROWS], np.int8),
                            ctx=np.array([r['ctx'] for r in ROWS], np.int8))
        log('collect saved %s' % os.path.basename(cache))
    npz_sha = hashlib.sha256(open(cache, 'rb').read()).hexdigest()[:8]
    del model
    torch.cuda.empty_cache()
    log('model released; npz sha8=%s' % npz_sha)

    # ---- 分析（CPU） ----
    z = np.load(cache)
    H16 = z['H'].astype(np.float32)
    EI = z['ei'].astype(int)
    RL = z['rel'].astype(int)
    PL = z['pol'].astype(int)
    CX = z['ctx'].astype(int)
    IDX = {(int(EI[i]), int(RL[i]), int(PL[i]), int(CX[i])): i for i in range(NROWS)}
    for ei in range(N_E):
        for rl in range(2):
            for pl in range(2):
                for cx in range(2):
                    assert (ei, rl, pl, cx) in IDX, ('missing cell', ei, rl, pl, cx)
    log('panel complete %d cells' % len(IDX))

    def orthonorm(M):
        U, s, Vt = np.linalg.svd(M, full_matrices=False)
        keep = s > 1e-9
        return U[:, keep]

    def seq_shares(X, blocks):
        """序贯投影份额; blocks = [(name, M_cols)]; 返回 dict + residual"""
        total = float((X ** 2).sum()) + 1e-18
        cum = np.zeros((X.shape[0], 0))
        shares = {}
        for nm, Bm in blocks:
            B = Bm.astype(np.float64)
            if B.shape[1] == 0:
                shares[nm] = 0.0
                continue
            B = B - cum @ (np.linalg.lstsq(cum, B, rcond=None)[0]) if cum.shape[1] else B
            Qb = orthonorm(B)
            if Qb.shape[1] == 0:
                shares[nm] = 0.0
                continue
            P = Qb @ (Qb.T @ X)
            shares[nm] = float((P ** 2).sum() / total)
            cum = np.hstack([cum, Qb])
        shares['residual'] = max(0.0, 1.0 - sum(shares.values()))
        return shares

    def onehot(vals, n):
        M = np.zeros((len(vals), n))
        for i, v in enumerate(vals):
            M[i, v] = 1.0
        return M

    def contrast(vals, levels):
        return np.array([[1.0 if v == a else -1.0 if v == b else 0.0 for v in vals]
                         for a, b in ((0, 1),)], dtype=np.float64).T

    def anova_at(k):
        X = H16[:, k, :].astype(np.float64)
        X = X - X.mean(0)
        blocks = [('E', onehot(EI.tolist(), N_E)),
                  ('R', contrast(RL.tolist(), 2)),
                  ('N', contrast(PL.tolist(), 2)),
                  ('C', contrast(CX.tolist(), 2)),
                  ('ExR', onehot([(ei * 2 + rl) for ei, rl in zip(EI.tolist(), RL.tolist())], N_E * 2) *
                   np.tile(contrast(RL.tolist(), 2), (1, 1)))]
        # ExR 需逐实体内对比: 直接构造交互 onehot
        er = np.zeros((NROWS, N_E))
        for i in range(NROWS):
            er[i, EI[i]] = 1.0 if RL[i] == 0 else -1.0
        blocks[4] = ('ExR', er)
        en = np.zeros((NROWS, N_E))
        ec = np.zeros((NROWS, N_E))
        for i in range(NROWS):
            en[i, EI[i]] = 1.0 if PL[i] == 0 else -1.0
            ec[i, EI[i]] = 1.0 if CX[i] == 1 else -1.0
        rcol = np.array([1.0 if RL[i] == 0 else -1.0 for i in range(NROWS)])
        ncol = np.array([1.0 if PL[i] == 0 else -1.0 for i in range(NROWS)])
        ccol = np.array([1.0 if CX[i] == 1 else -1.0 for i in range(NROWS)])
        rxn = (rcol * ncol).reshape(NROWS, 1)
        rxc = (rcol * ccol).reshape(NROWS, 1)
        nxc = (ncol * ccol).reshape(NROWS, 1)
        blocks += [('ExN', en), ('ExC', ec), ('RxN', rxn), ('RxC', rxc), ('NxC', nxc)]
        return seq_shares(X, blocks)

    sh_kout = anova_at(KOUT)
    sh_kstar = anova_at(KSTAR)
    log('ANOVA KOUT: ' + ' '.join('%s=%.4f' % (kk, v) for kk, v in sh_kout.items()))
    rxn_curve = {}
    for k in range(NH):
        rxn_curve[str(k)] = anova_at(k)['RxN']
    rxn_argmax = int(max(range(NH), key=lambda k: rxn_curve[str(k)]))

    # ---- 对易子 ----
    def commutators(k, cx):
        dR_p, dR_m = [], []
        dN_i, dN_h = [], []
        for ei in range(N_E):
            h_isa_p = H16[IDX[(ei, 0, 0, cx)], k]
            h_has_p = H16[IDX[(ei, 1, 0, cx)], k]
            h_isa_m = H16[IDX[(ei, 0, 1, cx)], k]
            h_has_m = H16[IDX[(ei, 1, 1, cx)], k]
            dR_p.append(h_isa_p - h_has_p)
            dR_m.append(h_isa_m - h_has_m)
            dN_i.append(h_isa_m - h_isa_p)
            dN_h.append(h_has_m - h_has_p)
        dR_p, dR_m = np.array(dR_p), np.array(dR_m)
        dN_i, dN_h = np.array(dN_i), np.array(dN_h)
        num_r = float(np.linalg.norm((dR_p - dR_m).ravel()))
        den_r = 0.5 * float(np.linalg.norm(dR_p.ravel()) + np.linalg.norm(dR_m.ravel())) + 1e-18
        num_n = float(np.linalg.norm((dN_i - dN_h).ravel()))
        den_n = 0.5 * float(np.linalg.norm(dN_i.ravel()) + np.linalg.norm(dN_h.ravel())) + 1e-18
        return num_r / den_r, num_n / den_n

    exch = {}
    for cx in (0, 1):
        er, en_ = commutators(KOUT, cx)
        exch['exchR_c%d' % cx] = er
        exch['exchN_c%d' % cx] = en_
        log('exch C=%d: R=%.4f N=%.4f' % (cx, er, en_))
    exchR_curve = {}
    for k in range(NH):
        er, _ = commutators(k, 0)
        exchR_curve[str(k)] = er

    # shuffle null (KOUT, C=0)
    rng = np.random.default_rng(3157)
    eperm_base = EI.copy()
    null_max = []
    for _ in range(N_SHUFFLE):
        perm = rng.permutation(N_E)
        EI_s = perm[EI]
        IDX_s = {(int(EI_s[i]), int(RL[i]), int(PL[i]), int(CX[i])): i for i in range(NROWS)}
        dR_p, dR_m = [], []
        for ei in range(N_E):
            dR_p.append(H16[IDX_s[(ei, 0, 0, 0)], KOUT] - H16[IDX_s[(ei, 1, 0, 0)], KOUT])
            dR_m.append(H16[IDX_s[(ei, 0, 1, 0)], KOUT] - H16[IDX_s[(ei, 1, 1, 0)], KOUT])
        num = float(np.linalg.norm((np.array(dR_p) - np.array(dR_m)).ravel()))
        den = 0.5 * float(np.linalg.norm(np.array(dR_p).ravel()) + np.linalg.norm(np.array(dR_m).ravel())) + 1e-18
        null_max.append(num / den)
    null95 = float(np.quantile(null_max, 0.95))
    exch_obs = max(exch['exchR_c0'], exch['exchN_c0'])
    log('shuffle null95=%.4f obs=%.4f' % (null95, exch_obs))

    # T_C 关系无关性
    dC = []
    for ei in range(N_E):
        for rl in range(2):
            for pl in range(2):
                dC.append(H16[IDX[(ei, rl, pl, 1)], KOUT] - H16[IDX[(ei, rl, pl, 0)], KOUT])
    dC = np.array(dC)
    dCn = dC / (np.linalg.norm(dC, axis=1, keepdims=True) + 1e-18)
    cosM = dCn @ dCn.T
    tri = cosM[np.triu_indices(len(dC), 1)]
    tc_cos = float(tri.mean())
    tc_norm_ratio = float(np.linalg.norm(dC.ravel()) /
                         (np.linalg.norm(H16[:, KOUT, :].astype(np.float64).ravel()) / np.sqrt(NROWS) * np.sqrt(len(dC)) + 1e-18))
    log('T_C: mean pairwise cos=%.4f' % tc_cos)

    # 门
    if exch_obs < EXCH_GATE_LO:
        comm_class = 'commutative_algebra_supported'
    elif exch_obs <= EXCH_GATE_HI:
        comm_class = 'commutative_partial'
    else:
        comm_class = 'non_commutative'
    sig = bool(exch_obs > null95)
    log('comm_class=%s significant_vs_null=%s' % (comm_class, sig))

    verdict = 'g2p2_%s|exchR_%.3f|exchN_%.3f|rxn_%.4f|tc_cos_%.3f|null95_%.3f' % (
        comm_class, exch['exchR_c0'], exch['exchN_c0'], sh_kout['RxN'], tc_cos, null95)
    if SMOKE:
        verdict = 'SMOKE_' + verdict

    result = dict(phase=3157, name=NAME, model=MODEL, smoke=SMOKE,
                  design_sha=exe_sha, nl=NL, hidden=D, readout=KOUT, n_rows=NROWS,
                  n_ent=N_E, runtime_s=round(time.time() - T0, 1),
                  determinism_note=det_note, npz_sha8=npz_sha,
                  anova_kout=sh_kout, anova_kstar=sh_kstar,
                  rxn_curve=rxn_curve, rxn_argmax_layer=rxn_argmax,
                  exch=exch, exchR_curve=exchR_curve,
                  shuffle_null95=null95, exch_significant=sig,
                  tc_mean_cos=tc_cos, tc_norm_ratio=tc_norm_ratio,
                  gates=dict(comm_class=comm_class, exchange_obs=exch_obs,
                             gate_lo=EXCH_GATE_LO, gate_hi=EXCH_GATE_HI),
                  verdict=verdict)
    res_sha8, seal = seal_result(result, 'result.json')
    log('DONE runtime=%.1fs' % (time.time() - T0))

# ---------------- summary ----------------
elif MODEL == 'summary':
    RBASE = os.path.join(RDIR, 'phase3157', NAME)
    MODELS = ['qwen3-4b', 'qwen3-14b', 'glm4']
    rs = {}
    for m in MODELS:
        rs[m] = json.load(open(os.path.join(RBASE, m, 'result.json'), encoding='utf-8'))
    SHARE_KEYS = ['E', 'R', 'N', 'C', 'ExR', 'ExN', 'ExC', 'RxN', 'RxC', 'NxC', 'residual']

    def vec(m, layer_key):
        d = rs[m][layer_key]
        return np.array([d[kk] for kk in SHARE_KEYS], float)

    v_kout = {m: vec(m, 'anova_kout') for m in MODELS}
    fp = {}
    for a in range(len(MODELS)):
        for b in range(a + 1, len(MODELS)):
            ma, mb = MODELS[a], MODELS[b]
            pk = float(np.corrcoef(v_kout[ma], v_kout[mb])[0, 1])
            ps = float(np.corrcoef(vec(ma, 'anova_kstar'), vec(mb, 'anova_kstar'))[0, 1])
            min_nl = min(rs[ma]['nl'], rs[mb]['nl'])
            pe = float(np.corrcoef([rs[ma]['exchR_curve'][str(k)] for k in range(min_nl)],
                                   [rs[mb]['exchR_curve'][str(k)] for k in range(min_nl)])[0, 1])
            fp['%s_vs_%s' % (ma, mb)] = dict(pearson_kout=pk, pearson_kstar=ps, pearson_exchR_curve=pe)
    fpmin = min(v['pearson_kout'] for v in fp.values())
    fp_pass = bool(fpmin >= FP_GATE)
    exch_mean = float(np.mean([rs[m]['gates']['exchange_obs'] for m in MODELS]))
    classes = set(rs[m]['gates']['comm_class'] for m in MODELS)
    if len(classes) == 1:
        comm_class = classes.pop()
    else:
        comm_class = 'mixed_' + '_'.join(sorted(c.split('_')[0] for c in classes))
    verdict = 'g2p2_%s|fpmin_%.3f|exch_mean_%.3f' % (comm_class, fpmin, exch_mean)
    if not fp_pass:
        verdict = verdict + '|fp_FAIL_desc'
    result = dict(phase=3157, name=NAME, model='summary',
                  fp_pairs=fp, fpmin_kout=fpmin, fp_pass=fp_pass,
                  shares_mean_kout={kk: float(np.mean([v_kout[m][i] for m in MODELS]))
                                    for i, kk in enumerate(SHARE_KEYS)},
                  exch_per_model={m: rs[m]['exch'] for m in MODELS},
                  exch_mean_obs=exch_mean, comm_class=comm_class,
                  per_model_verdicts={m: rs[m]['verdict'] for m in MODELS},
                  gates=dict(fp_gate=FP_GATE, fp_pass=fp_pass),
                  verdict=verdict)
    res_sha8, seal = seal_result(result, 'result_summary.json')
    log('SUMMARY DONE')
