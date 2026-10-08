# -*- coding: utf-8 -*-
# Phase 3155 (G2-P1): 多关系族与算子可分离性（K2 死线, 原 3154 顺延）
# 预注册: AGI_GPT5_MEMO 2026-10-05（观测前冻结）:
#   (1) 新面板采集: 3 关系族(is-a/has-a/made-of) x (实体 x 类) x 3 模板, 三模型全隐层(3151 协议)
#   (2) K2 检验: h(i,c) = W v_i + phi(c) 可分离性 —— 共享实体平衡面板 ANOVA
#       [E | C | T | ExC | ExT | CxT | ECT], 交互份额>50% -> 弃"条件门"独立结构(死线);
#       双向消融 = 实体读出跨关系迁移 + 关系读出跨实体迁移; 子空间角 = 三族实体子空间两两主角
#   (3) held-out 关系泛化门: 2 关系训练 -> 第 3 关系预测(err <= 1.5 x in-relation), 3 折轮换
#       迁移预测 piv(E,c*) = piv(E,c_tr) + mean_Cal[piv(.,c*) - piv(.,c_tr)]
#   (4) 验收: held-out 门 3 模型 + K2 份额表; held-out 全败 -> G2 降级描述学
# 材料: 24 共享实体(三族句均自然) + 每族 16 独有对 + 每族 6 held-out 实体对; 每对 3 模板
#   正式 = (24x3 + 16x3 + 6x3) 对 x 3 模板 = 414 行/模型; SMOKE = 共享 6 + 每族 extra 2 + ho 1 = 27 对 x 3 = 81 行
# 采集: last-token 全隐层 H fp16 (NROWS, NL+1, D); 确定性锚 3 行位级
# 判定层: KOUT=NL-1 (读出层, 3152 K1 判定层先例); KSTAR=round(0.075 NL); KGRID=[.375,.575,.775,.975]xNL
# 教训内置: SMOKE 目录分离; fail-fast 断言; numpy2 无 batch solve; PCA r=512 后 ridge(D 维直接 solve 太慢)
import os, sys, json, time, hashlib
import numpy as np

T0 = time.time()
try:
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
except Exception:
    pass
MODEL = os.environ.get('P3155_MODEL') or (sys.argv[1] if len(sys.argv) > 1 else 'qwen3-4b')
SMOKE = os.environ.get('P3155_SMOKE') == '1'
ROOT = r'D:\AI2050\Ai2050-OpenOne'
RDIR = os.path.join(ROOT, 'tests', 'glm5', 'result', 'rdc_query_construction_20260913')
NAME = 'g2p1_relation_family_operator_separability'
BASE = os.path.join(RDIR, 'phase3155', NAME, MODEL)
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
        json.dump({'phase': 3155, 'name': phase_name, 'design_sha': sha,
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
RELS = ['isa', 'hasa', 'mof']
TPL = {
    'isa':  ['{E}是一种{C}。', '{E}属于{C}。', '通常认为{E}是一种{C}。'],
    'hasa': ['{E}有{P}。', '{E}带有{P}。', '{E}都有{P}。'],
    'mof':  ['{E}由{M}制成。', '{E}是用{M}做的。', '{M}可以用来制作{E}。'],
}
# SHARED: (实体, is-a 上位类, has-a 部件, made-of 材料) —— 三族句均自然
SHARED = [
    ('桌子', '家具', '桌腿', '木头'),
    ('椅子', '座椅', '椅背', '木头'),
    ('窗户', '建筑构件', '窗框', '玻璃'),
    ('门', '建筑构件', '门把手', '木头'),
    ('硬币', '货币', '图案', '金属'),
    ('轮胎', '橡胶配件', '花纹', '橡胶'),
    ('纸', '材料', '纤维', '木浆'),
    ('书', '出版物', '书页', '纸张'),
    ('刀', '工具', '刀刃', '钢铁'),
    ('碗', '餐具', '碗底', '陶瓷'),
    ('衣服', '服饰', '袖子', '棉布'),
    ('鞋', '日用品', '鞋底', '皮革'),
    ('桥', '建筑', '桥墩', '钢筋混凝土'),
    ('房子', '建筑', '屋顶', '砖块'),
    ('汽车', '交通工具', '车轮', '钢铁'),
    ('钟表', '计时工具', '表盘', '金属'),
    ('杯子', '容器', '杯柄', '陶瓷'),
    ('瓶子', '容器', '瓶盖', '玻璃'),
    ('帽子', '服饰', '帽檐', '毛毡'),
    ('篮子', '容器', '提手', '竹条'),
    ('灯笼', '照明用具', '骨架', '薄纸'),
    ('锁', '保安用具', '锁孔', '金属'),
    ('钥匙', '五金用品', '齿纹', '铜'),
    ('伞', '雨具', '伞骨', '防水布'),
]
EXTRA = {
    'isa':  [('苹果', '水果'), ('玫瑰', '花卉'), ('金鱼', '观赏鱼'), ('老虎', '猛兽'),
             ('大象', '哺乳动物'), ('水稻', '粮食作物'), ('钻石', '宝石'), ('钢琴', '乐器'),
             ('冰箱', '家电'), ('火箭', '飞行器'), ('蜜蜂', '昆虫'), ('骆驼', '沙漠动物'),
             ('蘑菇', '真菌'), ('海豚', '海洋动物'), ('竹子', '禾本科植物'), ('鲸', '哺乳动物')],
    'hasa': [('苹果', '果核'), ('玫瑰', '刺'), ('大象', '长鼻'), ('老虎', '斑纹'),
             ('鸟', '翅膀'), ('树', '根系'), ('猫', '胡须'), ('鱼', '鱼鳍'),
             ('花', '花蕊'), ('山', '山脊'), ('河', '支流'), ('云', '水汽'),
             ('吉他', '琴弦'), ('书架', '隔板'), ('蜂巢', '蜂房'), ('南瓜', '瓜瓤')],
    'mof':  [('蜡烛', '蜡'), ('铁锅', '铸铁'), ('剪刀', '不锈钢'), ('口罩', '无纺布'),
             ('铅笔', '石墨'), ('粉笔', '石膏'), ('气球', '乳胶'), ('沙发', '海绵'),
             ('床垫', '弹簧'), ('凉席', '竹片'), ('毛笔', '狼毫'), ('砚台', '石头'),
             ('琴弦', '钢丝'), ('砧板', '竹木'), ('坛子', '陶土'), ('风铃', '玻璃')],
}
HO_E = {
    'isa':  [('莲花', '水生植物'), ('盾牌', '防御器械'), ('瀑布', '自然景观'),
             ('蟹', '甲壳动物'), ('火山', '地质景观'), ('银杏', '裸子植物')],
    'hasa': [('椰子', '硬壳'), ('菠萝', '冠芽'), ('蜘蛛', '八条腿'),
             ('钟楼', '大钟'), ('城堡', '塔楼'), ('竹子', '竹节')],
    'mof':  [('木船', '木板'), ('石桥', '花岗岩'), ('草帽', '麦秆'),
             ('皮包', '牛皮'), ('瓷碗', '高岭土'), ('竹筏', '毛竹')],
}
N_SH = 6 if SMOKE else 24
N_EX = 2 if SMOKE else 16
N_HO = 1 if SMOKE else 6
K2_GATE = 0.5          # 死线: KOUT 层 ExC 交互份额 > 0.5 -> 弃条件门
HO_GATE_RATIO = 1.5    # held-out 迁移 err / in-relation 基线 err 门
FP_GATE = 0.8
TOPR_SUB = 8           # 每族实体子空间维数
RIDGE_PCA = 512        # 双向消融 ridge 前的 PCA 维数

def tpl_sent(rel, e, x, t):
    return TPL[rel][t].replace('{E}', e).replace('{C}', x).replace('{P}', x).replace('{M}', x)

# ---- fail-fast 材料断言（任何 GPU 观测之前） ----
def material_asserts():
    notes = []
    for rel in RELS:
        assert len(TPL[rel]) == 3 and len(set(TPL[rel])) == 3, ('tpl distinct', rel)
    for k, tup in enumerate(SHARED[:N_SH]):
        assert len(tup) == 4 and all(isinstance(x, str) and x for x in tup), ('shared tuple', k)
        for ri, rel in enumerate(RELS):
            x = tup[1 + ri]
            for t in range(3):
                s = tpl_sent(rel, tup[0], x, t)
                assert tup[0] in s and x in s, ('literal missing', rel, tup[0], x, t)
    for rel in RELS:
        for (e, x) in EXTRA[rel][:N_EX] + HO_E[rel][:N_HO]:
            for t in range(3):
                s = tpl_sent(rel, e, x, t)
                assert e in s and x in s, ('literal missing extra', rel, e, x, t)
    notes.append('shared=%d extra=%d/family ho=%d/family tpl=3/family' % (N_SH, N_EX, N_HO))
    return notes

MAT_NOTES = material_asserts()

# ---------------- 行表构建 ----------------
def build_rows():
    rows = []
    for k in range(N_SH):
        e, c, p, m = SHARED[k]
        for ri, rel in enumerate(RELS):
            x = (c, p, m)[ri]
            for t in range(3):
                rows.append(dict(rel=ri, reln=rel, ent=e, cls=x, tmpl=t, ent_id=k,
                                 src='shared', prompt=tpl_sent(rel, e, x, t), ho=0))
    base_id = N_SH
    for ri, rel in enumerate(RELS):
        arr = EXTRA[rel][:N_EX]
        for j, (e, x) in enumerate(arr):
            for t in range(3):
                rows.append(dict(rel=ri, reln=rel, ent=e, cls=x, tmpl=t, ent_id=base_id + j,
                                 src='extra', prompt=tpl_sent(rel, e, x, t), ho=0))
        base_id += len(arr)
    for ri, rel in enumerate(RELS):
        arr = HO_E[rel][:N_HO]
        for j, (e, x) in enumerate(arr):
            for t in range(3):
                rows.append(dict(rel=ri, reln=rel, ent=e, cls=x, tmpl=t, ent_id=100 + j,
                                 src='hoent', prompt=tpl_sent(rel, e, x, t), ho=1))
    return rows

ROWS = build_rows()
NROWS = len(ROWS)
NTR = sum(1 for r in ROWS if r['ho'] == 0)
NHO = NROWS - NTR
EXP_TR = (6 * 3 + 3 * 2) * 3 if SMOKE else (24 * 3 + 3 * 16) * 3
EXP_HO = 3 * 1 * 3 if SMOKE else 3 * 6 * 3
assert NTR == EXP_TR and NHO == EXP_HO, ('panel size', NTR, NHO, EXP_TR, EXP_HO)
log('panel rows=%d (train %d + hoent %d), shared=%d, smoke=%s' %
    (NROWS, NTR, NHO, N_SH, SMOKE))

# ---------------- 正交工具（3154 复用） ----------------
def orthonorm(M, drop_tol=1e-8):
    U, s, Vt = np.linalg.svd(np.asarray(M, dtype=np.float64), full_matrices=False)
    keep = s > (s[0] * drop_tol if s.size and s[0] > 0 else 1e-12)
    return U[:, keep]

def orth_against(M, *qlist):
    R = np.asarray(M, dtype=np.float64).copy()
    for Q in qlist:
        R = R - Q @ (Q.T @ R)
    return R

# ---------------- K2 平衡面板基（共享实体子面板） ----------------
SH_IDX = [i for i, r in enumerate(ROWS) if r['src'] == 'shared']

def build_k2_bases():
    n = len(SH_IDX)
    n_ent = N_SH
    E = np.zeros((n, n_ent)); C = np.zeros((n, 3)); T = np.zeros((n, 3))
    EC = np.zeros((n, n_ent * 3)); ET = np.zeros((n, n_ent * 3)); CT = np.zeros((n, 9))
    for j, i in enumerate(SH_IDX):
        r = ROWS[i]
        E[j, r['ent_id']] = 1.0; C[j, r['rel']] = 1.0; T[j, r['tmpl']] = 1.0
        EC[j, r['ent_id'] * 3 + r['rel']] = 1.0
        ET[j, r['ent_id'] * 3 + r['tmpl']] = 1.0
        CT[j, r['rel'] * 3 + r['tmpl']] = 1.0
    QE = orthonorm(E)
    QC = orthonorm(orth_against(orthonorm(C), QE))
    QT = orthonorm(orth_against(orthonorm(T), QE, QC))
    QEC = orthonorm(orth_against(orthonorm(EC), QE, QC, QT))
    QET = orthonorm(orth_against(orthonorm(ET), QE, QC, QT))
    QCT = orthonorm(orth_against(orthonorm(CT), QE, QC, QT))
    expd = dict(QE=(n, n_ent), QC=(n, 2), QT=(n, 2),
                QEC=(n, (n_ent - 1) * 2), QET=(n, (n_ent - 1) * 2), QCT=(n, 4))
    for nm, Q in (('QE', QE), ('QC', QC), ('QT', QT), ('QEC', QEC), ('QET', QET), ('QCT', QCT)):
        assert Q.shape == expd[nm], ('basis rank', nm, Q.shape, expd[nm])
    names = ['QE', 'QC', 'QT', 'QEC', 'QET', 'QCT']
    bmap = {'QE': QE, 'QC': QC, 'QT': QT, 'QEC': QEC, 'QET': QET, 'QCT': QCT}
    # 两两正交断言（平衡设计应严格成立）
    for a in range(len(names)):
        for b in range(a + 1, len(names)):
            xo = float(np.abs(bmap[names[a]].T @ bmap[names[b]]).max())
            assert xo < 1e-8, ('basis not orthogonal', names[a], names[b], xo)
    return dict(QE=QE, QC=QC, QT=QT, QEC=QEC, QET=QET, QCT=QCT,
                key={'E': 'QE', 'C': 'QC', 'T': 'QT', 'EintC': 'QEC',
                     'EintT': 'QET', 'CintT': 'QCT'})

def k2_shares_at(Hk, B):
    X = Hk[SH_IDX].astype(np.float64)
    R0 = X - X.mean(0)
    SS = float((R0 ** 2).sum())
    if SS < 1e-10:
        # embedding 层 last-token 恒为句号(RoPE 模型位置信息不进 h0) -> 零方差, 份额无定义记 0
        return {k: 0.0 for k in ('E', 'C', 'T', 'EintC', 'EintT', 'CintT', 'ECT')}
    SS += 1e-18
    e = {}
    resid = R0
    for nm in ('E', 'C', 'T', 'EintC', 'EintT', 'CintT'):
        Q = B[B['key'][nm]]
        P = Q @ (Q.T @ resid)
        e[nm] = float((P ** 2).sum())
        resid = resid - P
    e['ECT'] = float((resid ** 2).sum())   # 饱和剩余 = 三因素交互(每格 1 观测)
    sh = {k: v / SS for k, v in e.items()}
    assert abs(sum(sh.values()) - 1.0) < 1e-6, ('k2 anova closure', sum(sh.values()))
    return sh

# ---------------- 双向消融（ridge one-hot, PCA r=512） ----------------
def ridge_fit_predict(Ztr, Ytr, Zte, lam=1e-3):
    D = Ztr.shape[1]
    W = np.linalg.solve(Ztr.T @ Ztr + lam * np.eye(D), Ztr.T @ Ytr)
    return Zte @ W

def onehot(ids, k):
    Y = np.zeros((len(ids), k))
    for j, v in enumerate(ids):
        Y[j, v] = 1.0
    return Y

def bidirectional_ablation(Hk, B):
    """实体读出跨关系迁移 + 关系读出跨实体迁移（PCA 降维 ridge, 返回各层位 acc 表）"""
    Xs = Hk[SH_IDX].astype(np.float32)
    mu = Xs.mean(0)
    U, s, Vt = np.linalg.svd(Xs - mu, full_matrices=False)
    r = min(RIDGE_PCA, Vt.shape[0])
    V = Vt[:r].T                      # (D, r) D 空间右奇异基
    en_keep = float((s[:r] ** 2).sum() / (s ** 2).sum() + 1e-18)
    Z = ((Xs - mu) @ V).astype(np.float64)   # (n, r)
    rel_of = np.array([ROWS[i]['rel'] for i in SH_IDX])
    ent_of = np.array([ROWS[i]['ent_id'] for i in SH_IDX])
    tmpl_of = np.array([ROWS[i]['tmpl'] for i in SH_IDX])
    out = dict(pca_r=int(r), energy_kept=en_keep)
    # 实体读出: train 族 c1 全行 -> test 族 c2 同实体行
    ent_acc = {}
    for c1 in range(3):
        tr = np.where(rel_of == c1)[0]
        Ytr = onehot(list(ent_of[tr]), N_SH)
        W = np.linalg.solve(Z[tr].T @ Z[tr] + 1e-3 * np.eye(Z.shape[1]), Z[tr].T @ Ytr)
        for c2 in range(3):
            if c2 == c1:
                continue
            te = np.where(rel_of == c2)[0]
            pred = np.argmax(Z[te] @ W, 1)
            ent_acc['%d->%d' % (c1, c2)] = float(np.mean(pred == ent_of[te]))
        # within: 留一模板轮换（训练含全部实体, 测试同实体异模板）
        accs = []
        for ht in range(3):
            trn = tr[tmpl_of[tr] != ht]
            tst = tr[tmpl_of[tr] == ht]
            Wf = np.linalg.solve(Z[trn].T @ Z[trn] + 1e-3 * np.eye(Z.shape[1]),
                                 Z[trn].T @ onehot(list(ent_of[trn]), N_SH))
            accs.append(float(np.mean(np.argmax(Z[tst] @ Wf, 1) == ent_of[tst])))
        ent_acc['within_c%d' % c1] = float(np.mean(accs))
    out['entity_readout'] = ent_acc
    # 关系读出: train 实体前半 -> test 实体后半
    half = N_SH // 2
    tr = np.where(ent_of < half)[0]
    te = np.where(ent_of >= half)[0]
    W = np.linalg.solve(Z[tr].T @ Z[tr] + 1e-3 * np.eye(Z.shape[1]),
                        Z[tr].T @ onehot(list(rel_of[tr]), 3))
    rel_acc_cross = float(np.mean(np.argmax(Z[te] @ W, 1) == rel_of[te]))
    idx = np.arange(len(tr))
    rng = np.random.RandomState(0)
    rng.shuffle(idx)
    a1, a2 = idx[::2], idx[1::2]
    Wf = np.linalg.solve(Z[tr[a1]].T @ Z[tr[a1]] + 1e-3 * np.eye(Z.shape[1]),
                         Z[tr[a1]].T @ onehot(list(rel_of[tr[a1]]), 3))
    rel_acc_within = float(np.mean(np.argmax(Z[tr[a2]] @ Wf, 1) == rel_of[tr[a2]]))
    out['relation_readout'] = dict(cross_entity=rel_acc_cross, within_entity=rel_acc_within)
    return out

# ---------------- 子空间角 ----------------
def subspace_angles(Hk):
    """三族实体子空间两两主角 + 实体合并子空间 vs 关系子空间"""
    piv = np.zeros((N_SH, 3, Hk.shape[1]))
    for j, i in enumerate(SH_IDX):
        r = ROWS[i]
        piv[r['ent_id'], r['rel']] += Hk[i].astype(np.float64) / 3.0
    out = {}
    pcs = {}
    for c in range(3):
        M = piv[:, c, :] - piv[:, c, :].mean(0)
        _, s, Vt = np.linalg.svd(M, full_matrices=False)
        r = min(TOPR_SUB, Vt.shape[0])
        pcs[c] = Vt[:r].T                 # (D, r) D 空间散布主方向
    pair_sv = {}
    for a, b in ((0, 1), (0, 2), (1, 2)):
        sv = np.linalg.svd(pcs[a].T @ pcs[b], compute_uv=False)
        pair_sv['%d_vs_%d' % (a, b)] = dict(top1=float(sv[0]), mean8=float(sv.mean()))
    out['entity_subspace_pairs'] = pair_sv
    tall = piv.reshape(N_SH * 3, -1)
    rel_seq = np.tile(np.arange(3), N_SH)          # reshape 行序 ent-major -> rel = idx % 3
    cm0 = piv.mean(0)                              # (3, D) 每族跨实体均值
    cm = cm0 - cm0.mean(0)
    _, s2, Vt2 = np.linalg.svd(cm, full_matrices=False)
    Q_rel = Vt2[:2].T                              # (D, 2) 关系子空间
    tall_c = tall - cm0[rel_seq]                   # 每质心剔所属族均值 -> 族内实体散布
    _, se, Vte = np.linalg.svd(tall_c, full_matrices=False)
    P_ent = Vte[:min(16, Vte.shape[0])].T          # (D, r)
    sv = np.linalg.svd(P_ent.T @ Q_rel, compute_uv=False)
    out['entity_vs_relation'] = dict(sv_max=float(sv.max()), sv_min=float(sv.min()),
                                     sv_mean=float(sv.mean()))
    return out

# ---------------- held-out 关系泛化（3 折轮换） ----------------
def heldout_relation_folds(Hk):
    """pivot=模板平均; fold held=c*: 迁移 err vs in-relation 基线 err"""
    piv = np.zeros((N_SH, 3, Hk.shape[1]))
    for j, i in enumerate(SH_IDX):
        r = ROWS[i]
        piv[r['ent_id'], r['rel']] += Hk[i].astype(np.float64) / 3.0
    half = N_SH // 2
    cal = list(range(half))
    test = list(range(half, N_SH))
    folds = {}
    for cstar in range(3):
        ctrs = [c for c in range(3) if c != cstar]
        preds = []
        ds = {}
        for ct in ctrs:
            d = piv[cal, cstar, :].mean(0) - piv[cal, ct, :].mean(0)
            ds[ct] = d
            preds.append(piv[test][:, ct, :] + d[None, :])
        pred_ens = (preds[0] + preds[1]) / 2.0
        err_trans = float(((piv[test][:, cstar, :] - pred_ens) ** 2).sum(1).mean())
        bases = []
        for ct in ctrs:
            mu_cal = piv[cal][:, ct, :].mean(0)
            bases.append(float(((piv[test][:, ct, :] - mu_cal[None, :]) ** 2).sum(1).mean()))
        err_base = float(np.mean(bases))
        ratio = err_trans / (err_base + 1e-18)
        folds['held_%s' % RELS[cstar]] = dict(
            err_trans=err_trans, err_base=err_base, ratio=ratio,
            pass_gate=bool(ratio <= HO_GATE_RATIO),
            per_source={str(ct): float(((piv[test][:, cstar, :] - (piv[test][:, ct, :] + ds[ct][None, :])) ** 2).sum(1).mean())
                        for ct in ctrs})
    return folds

# ---------------- 模式 A/B/C: GPU 采集 + 分析 ----------------
if MODEL in ('qwen3-4b', 'qwen3-14b', 'glm4'):
    MDIR_MAP = {'qwen3-4b': 'qwen3-4b', 'qwen3-14b': 'Qwen3-14B', 'glm4': 'glm4-9b-chat-hf'}
    MDIR = os.path.join(ROOT, 'models', 'hf', MDIR_MAP[MODEL])
    cfg = json.load(open(os.path.join(MDIR, 'config.json'), encoding='utf-8'))
    NL = cfg['num_hidden_layers']
    HID = cfg['hidden_size']
    KSTAR = int(round(0.075 * NL))
    KOUT = NL - 1
    KGRID = [int(round(f * NL)) for f in (0.375, 0.575, 0.775, 0.975)]
    design = dict(model=MODEL, mdir=MDIR, phase='g2p1_relation_family_operator_separability',
                  nl=NL, hidden=HID, kstar=KSTAR, readout=KOUT, kgrid=KGRID,
                  rels=RELS, templates=TPL, shared=SHARED[:N_SH],
                  extra={k: v[:N_EX] for k, v in EXTRA.items()},
                  ho_entities={k: v[:N_HO] for k, v in HO_E.items()},
                  n_rows=NROWS, n_train=NTR, n_hoent=NHO,
                  k2_gate=K2_GATE, ho_gate_ratio=HO_GATE_RATIO, fp_gate=FP_GATE,
                  topr_sub=TOPR_SUB, ridge_pca=RIDGE_PCA,
                  fold_cal='ent_id < N_SH//2', fold_test='ent_id >= N_SH//2',
                  material_asserts=MAT_NOTES, smoke=SMOKE,
                  pre_reg='MEMO 2026-10-05 (original G2-P1, deferred from 3154); frozen before observation')
    exe_sha = freeze_design('g2p1_%s' % MODEL, design)
    log('model=%s NL=%d D=%d k*=%d readout=%d rows=%d' % (MODEL, NL, HID, KSTAR, KOUT, NROWS))

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
                ii = torch.tensor([tok(r['prompt'], add_special_tokens=False)['input_ids']],
                                  device='cuda')
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
                ii = torch.tensor([tok(ROWS[i]['prompt'], add_special_tokens=False)['input_ids']],
                                  device='cuda')
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
                            rel=np.array([r['rel'] for r in ROWS], np.int8),
                            ent_id=np.array([r['ent_id'] for r in ROWS], np.int16),
                            tmpl=np.array([r['tmpl'] for r in ROWS], np.int8),
                            src=np.array([1 if r['src'] == 'shared' else (0 if r['src'] == 'extra' else 2)
                                          for r in ROWS], np.int8),
                            ho=np.array([r['ho'] for r in ROWS], np.int8))
        log('collect saved %s' % os.path.basename(cache))
    npz_sha = hashlib.sha256(open(cache, 'rb').read()).hexdigest()[:8]
    del model
    torch.cuda.empty_cache()
    log('model released; npz sha8=%s' % npz_sha)

    # ---- 分析（CPU） ----
    BK = build_k2_bases()
    def Hk(k):
        return H16[:, k, :].astype(np.float32)

    sh_kout = k2_shares_at(Hk(KOUT), BK)
    sh_kstar = k2_shares_at(Hk(KSTAR), BK)
    sh_emb = k2_shares_at(Hk(0), BK)
    log('k2 shares KOUT: ' + ' '.join('%s=%.4f' % (k, v) for k, v in sh_kout.items()))

    # 全层 ExC 交互份额曲线
    curve_int = {}
    for k in range(NH):
        curve_int[str(k)] = k2_shares_at(Hk(k), BK)['EintC']
    int_argmax = int(max(range(NH), key=lambda k: curve_int[str(k)]))

    # 双向消融（KOUT + KSTAR + KGRID）
    bid = {}
    for k in [KOUT, KSTAR] + KGRID:
        bid[str(k)] = bidirectional_ablation(Hk(k), BK)
        ba = bid[str(k)]
        log('ablation k=%d: ent cross mean=%.3f within=%.3f | rel cross=%.3f within=%.3f' % (
            k, float(np.mean([v for kk, v in ba['entity_readout'].items() if '->' in kk])),
            float(np.mean([v for kk, v in ba['entity_readout'].items() if 'within' in kk])),
            ba['relation_readout']['cross_entity'], ba['relation_readout']['within_entity']))

    # 子空间角（KOUT + KGRID）
    sub = {}
    for k in [KOUT] + KGRID:
        sub[str(k)] = subspace_angles(Hk(k))
        sv = sub[str(k)]
        log('subspace k=%d: ent pair top1=%s | ent-vs-rel sv_max=%.3f' % (
            k, {kk: round(vv['top1'], 3) for kk, vv in sv['entity_subspace_pairs'].items()},
            sv['entity_vs_relation']['sv_max']))

    # held-out 关系 fold（逐层）
    folds_out = {}
    for k in [KOUT] + KGRID:
        folds_out[str(k)] = heldout_relation_folds(Hk(k))
    ho_kout = folds_out[str(KOUT)]
    n_pass = sum(1 for v in ho_kout.values() if v['pass_gate'])
    for fn, fv in ho_kout.items():
        log('fold %s: ratio=%.3f pass=%s (err_trans=%.4f err_base=%.4f)' %
            (fn, fv['ratio'], fv['pass_gate'], fv['err_trans'], fv['err_base']))

    k2int_kout = sh_kout['EintC']
    ent_cross_mean = float(np.mean([v for kk, v in bid[str(KOUT)]['entity_readout'].items() if '->' in kk]))
    ent_within_mean = float(np.mean([v for kk, v in bid[str(KOUT)]['entity_readout'].items() if 'within' in kk]))
    rel_cross = bid[str(KOUT)]['relation_readout']['cross_entity']
    entxrel_sep = ent_cross_mean / (ent_within_mean + 1e-12)
    ratio_mean = float(np.mean([v['ratio'] for v in ho_kout.values()]))

    verdict = ('g2p1_%s|k2int_%.4f|entxrel_%.2f_vs_within_%.2f|relxent_%.2f|'
               'ho_ratio_mean_%.2f_pass_%d/3' %
               (MODEL, k2int_kout, ent_cross_mean, ent_within_mean, rel_cross,
                ratio_mean, n_pass))
    result = dict(
        phase=3155, name=NAME, mode=MODEL,
        created=time.strftime('%Y-%m-%d %H:%M:%S'),
        runtime_s=round(time.time() - T0, 1),
        design_sha=exe_sha, smoke=SMOKE,
        n_rows=NROWS, n_train=NTR, n_hoent=NHO, n_shared=N_SH,
        NL=NL, hidden=HID, kstar=KSTAR, readout=KOUT, kgrid=KGRID,
        npz=cache, npz_sha8=npz_sha, determinism_note=det_note,
        k2_shares_kout=sh_kout, k2_shares_kstar=sh_kstar, k2_shares_emb=sh_emb,
        curve_EintC=curve_int, curve_EintC_argmax_layer=int_argmax,
        fp_kout=[sh_kout[x] for x in ('E', 'C', 'EintC', 'T', 'EintT', 'CintT', 'ECT')],
        fp_kstar=[sh_kstar[x] for x in ('E', 'C', 'EintC', 'T', 'EintT', 'CintT', 'ECT')],
        bidirectional_ablation=bid, subspace_angles=sub,
        heldout_folds=folds_out,
        gates=dict(k2_gate=K2_GATE, k2_inseparable=bool(k2int_kout > K2_GATE),
                   ho_gate_ratio=HO_GATE_RATIO, ho_pass_count=n_pass,
                   ho_all_pass=bool(n_pass == 3)),
        honesty_notes=['模板 T 与句长部分混杂（本设计不做字数守恒）, T 份额解释需谨慎',
                       '类词/部件词/材料词逐实体唯一, 词频差异归入实体因素内',
                       'made-of 第三模板主语为材料词, 结构差异计入 T 因素'],
        grade='statistical',
        verdict=verdict)
    json.dump({'rows': [{k: r[k] for k in ('rel', 'reln', 'ent', 'cls', 'tmpl', 'ent_id',
                                           'src', 'prompt', 'ho')} for r in ROWS]},
              open(os.path.join(BASE, 'materials.json'), 'w', encoding='utf-8'),
              ensure_ascii=False, indent=1)
    seal_result(result, 'result.json')
    log('DONE mode=%s runtime %.1fs' % (MODEL, time.time() - T0))
    sys.exit(0)

# ---------------- 模式 D: summary —— K2 死线判决 + 指纹 ----------------
if MODEL == 'summary':
    P4B = os.path.join(RDIR, 'phase3155', NAME, 'qwen3-4b', 'result.json')
    P14B = os.path.join(RDIR, 'phase3155', NAME, 'qwen3-14b', 'result.json')
    PG4 = os.path.join(RDIR, 'phase3155', NAME, 'glm4', 'result.json')
    r4b = json.load(open(P4B, encoding='utf-8'))
    r14b = json.load(open(P14B, encoding='utf-8'))
    rg4 = json.load(open(PG4, encoding='utf-8'))
    design = dict(inputs={'qwen3-4b': dict(path=P4B, sha8=r4b.get('res_sha8'), seal=r4b.get('seal_sha8')),
                          'qwen3-14b': dict(path=P14B, sha8=r14b.get('res_sha8'), seal=r14b.get('seal_sha8')),
                          'glm4': dict(path=PG4, sha8=rg4.get('res_sha8'), seal=rg4.get('seal_sha8'))},
                  fp_gate=FP_GATE, k2_gate=K2_GATE, ho_gate_ratio=HO_GATE_RATIO,
                  pre_reg='MEMO 2026-10-05; K2 death-line + fingerprint on KOUT 7-dim shares')
    exe_sha = freeze_design('g2p1_summary', design)
    fps = {'qwen3-4b': r4b['fp_kout'], 'qwen3-14b': r14b['fp_kout'], 'glm4': rg4['fp_kout']}
    fps_ks = {'qwen3-4b': r4b['fp_kstar'], 'qwen3-14b': r14b['fp_kstar'], 'glm4': rg4['fp_kstar']}
    pairs = {}
    for a, b in (('qwen3-4b', 'qwen3-14b'), ('qwen3-4b', 'glm4'), ('qwen3-14b', 'glm4')):
        r = float(np.corrcoef(np.array(fps[a]), np.array(fps[b]))[0, 1])
        rk = float(np.corrcoef(np.array(fps_ks[a]), np.array(fps_ks[b]))[0, 1])
        pairs['%s_vs_%s' % (a, b)] = dict(pearson_kout=r, pearson_kstar=rk,
                                          pass_gate=bool(r >= FP_GATE))
        log('fp %s vs %s: kout r=%.4f kstar r=%.4f' % (a, b, r, rk))
    fpmin = min(p['pearson_kout'] for p in pairs.values())
    all_fp = all(p['pass_gate'] for p in pairs.values())
    k2ints = {m: rr['k2_shares_kout']['EintC'] for m, rr in
              (('qwen3-4b', r4b), ('qwen3-14b', r14b), ('glm4', rg4))}
    k2_sep_all = all(v <= K2_GATE for v in k2ints.values())
    ho_counts = {m: rr['gates']['ho_pass_count'] for m, rr in
                 (('qwen3-4b', r4b), ('qwen3-14b', r14b), ('glm4', rg4))}
    ho_total = sum(ho_counts.values())
    ho_all_models = all(v == 3 for v in ho_counts.values())
    ho_none = all(v == 0 for v in ho_counts.values())
    k2int_mean = float(np.mean(list(k2ints.values())))
    if ho_none:
        verdict = 'g2p1_ho_fail_g2_degrade_descriptive|k2int_%.4f|fpmin_%.3f' % (k2int_mean, fpmin)
    elif not k2_sep_all:
        verdict = ('g2p1_k2_inseparable_discard_conditional_gate|k2int_%.4f|fpmin_%.3f|'
                   'ho_pass_%d/9' % (k2int_mean, fpmin, ho_total))
    elif k2_sep_all and ho_all_models and all_fp:
        verdict = 'g2p1_k2_separable_conditional_gate_supported|k2int_%.4f|fpmin_%.3f|ho_pass_9/9' % (
            k2int_mean, fpmin)
    else:
        verdict = ('g2p1_mixed_partial|k2int_%.4f|fpmin_%.3f|ho_pass_%d/9|k2_sep_%s|fp_%s' %
                   (k2int_mean, fpmin, ho_total,
                    str(k2_sep_all), str(all_fp)))
    per_model = {}
    for mn, rr in (('qwen3-4b', r4b), ('qwen3-14b', r14b), ('glm4', rg4)):
        per_model[mn] = dict(
            NL=rr['NL'], kstar=rr['kstar'], readout=rr['readout'],
            k2_shares_kout=rr['k2_shares_kout'],
            k2_shares_kstar=rr['k2_shares_kstar'],
            curve_EintC_argmax=rr['curve_EintC_argmax_layer'],
            ent_cross=float(np.mean([v for kk, v in rr['bidirectional_ablation'][str(rr['readout'])]
                                     ['entity_readout'].items() if '->' in kk])),
            rel_cross=rr['bidirectional_ablation'][str(rr['readout'])]['relation_readout']['cross_entity'],
            subspace_ent_pair_top1={kk: vv['top1'] for kk, vv in
                                    rr['subspace_angles'][str(rr['readout'])]['entity_subspace_pairs'].items()},
            ent_vs_rel_sv_max=rr['subspace_angles'][str(rr['readout'])]['entity_vs_relation']['sv_max'],
            ho_fold_ratios={kk: vv['ratio'] for kk, vv in rr['heldout_folds'][str(rr['readout'])].items()},
            ho_pass=rr['gates']['ho_pass_count'],
            verdict=rr['verdict'])
    result = dict(
        phase=3155, name=NAME, mode='summary',
        created=time.strftime('%Y-%m-%d %H:%M:%S'),
        runtime_s=round(time.time() - T0, 1),
        design_sha=exe_sha, smoke=SMOKE,
        fp_pairs=pairs, fp_min_kout=fpmin, fingerprint_consistent=bool(all_fp),
        k2_int_share_kout=k2ints, k2_separable_all=bool(k2_sep_all),
        ho_pass_counts=ho_counts, ho_total=ho_total,
        k2_death_line_triggered=bool(not k2_sep_all),
        per_model=per_model,
        shares_mean_kout={k: float(np.mean([fps[m][i] for m in sorted(fps)]))
                          for i, k in enumerate(('E', 'C', 'EintC', 'T', 'EintT', 'CintT', 'ECT'))},
        inputs_used=design['inputs'],
        grade='statistical',
        verdict=verdict)
    seal_result(result, 'result_summary.json')
    log('DONE mode=summary runtime %.1fs' % (time.time() - T0))
    sys.exit(0)

log('unknown mode %s' % MODEL)
sys.exit(2)
