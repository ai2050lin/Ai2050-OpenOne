# -*- coding: utf-8 -*-
# Phase 3154 (G1-P4): 多因素混杂分解 —— 语言/风格/逻辑/标点距离四因素在隐状态中的可分离性
# 预注册：AGI_GPT5_MEMO（2026-10-01 观测前冻结；原 G2-P1 多关系族顺延为 Phase 3155）
# 运行模式（P3154_MODEL 环境变量 或 argv[1]）: qwen3-4b | qwen3-14b | glm4 | summary
# 设计（冻结）:
#   材料 = 6 主题(天气/运动/饮食/学习/植物/市场) x 2 语言(zh/en) x 2 风格(formal/casual)
#          x 2 逻辑(con/contra) x 4 同字数移逗号变体 = 192 句/模型
#   cut in {20%,45%,70%,90%}: zh 按字 c=round(frac*len(A)), prompt=A[:c]+','+A[c:]+B
#                             en 按词 w=round(frac*len(words)), prompt=pre+','+' '+suf+' '+B
#   字数守恒: 每组 4 变体字符数恒等（逗号只在 A 内部移位, B 不动）
#   con/contra 最小对: zh 字数相等, en 词数相等（逐条 assert, fail-fast 在任何 GPU 观测前）
#   held-out: T7 睡眠 / T8 冷链 x 4 语言风格组合 x 2 逻辑 x cut=45% = 16 句
#   SMOKE: 主题[:2] x cuts[:2] = 32 句 + held-out T7 8 句 = 40 行; null 方向 20
#   采集: last-token 全隐层 H fp16 (NROWS, NL+1, D); collect.npz sha8 登记
#   D 协变量(连续) = n_tok(prompt) - n_tok(无逗号前缀), 即逗号到句尾的 token 距离
#   确定性锚: 3 行重前向位级比对(fp16)
#   分析: (1) 序贯投影 ANOVA [L -> S -> content(24 块 dummy) -> G -> D(连续) -> resid]
#              份额和=1 (正交基预计算一次, 逐层矩阵乘); KOUT/KSTAR 全表 + 全层 G/L 曲线
#         (2) 指纹门: KOUT 6 维份额向量三模型两两 Pearson >= 0.8 (summary 模式, 3 对)
#             任一对 < 0.8 -> g1p4_fingerprint_inconsistent_material_method_descriptive
#         (3) 几何干预: w_G = 归一化组均值差(contra - con)@层;
#             flip = con 行注入 +0.5*分离度 后投影越 0 率; null = 100 随机单位方向同流程
#         (4) held-out 门: w_G 于 16 句符号准确率 >= 0.75
# 教训内置: numpy2 无 solve batch 需求; f-string 避 %% 陷阱; SMOKE 目录分离; 断言 fail-fast
import os, sys, json, time, hashlib
import numpy as np

T0 = time.time()
try:
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
except Exception:
    pass
MODEL = os.environ.get('P3154_MODEL') or (sys.argv[1] if len(sys.argv) > 1 else 'qwen3-4b')
SMOKE = os.environ.get('P3154_SMOKE') == '1'
ROOT = r'D:\AI2050\Ai2050-OpenOne'
RDIR = os.path.join(ROOT, 'tests', 'glm5', 'result', 'rdc_query_construction_20260913')
NAME = 'g1p4_mfd_multifactor_disentangle'
BASE = os.path.join(RDIR, 'phase3154', NAME, MODEL)
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
        json.dump({'phase': 3154, 'name': phase_name, 'design_sha': sha,
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
TOPICS = ['天气', '运动', '饮食', '学习', '植物', '市场']
TOPICS_HO = ['睡眠', '冷链']
LANGS = ['zh', 'en']
STYLES = ['formal', 'casual']
LOGICS = ['con', 'contra']
CUT_FRACS = (0.20, 0.45, 0.70, 0.90)
HO_CUT_IDX = 1                      # held-out 固定 cut=45%
NULL_DIRS = 20 if SMOKE else 100
FP_GATE = 0.8
HO_GATE = 0.75
ALPHA_FRAC = 0.5
MONO_MAX_VIOL = 0.5                 # d 单调违例组比例上限

# MAT[(topic, lang, style)] = (A, B_con, B_contra)
MAT = {
    ('天气', 'zh', 'formal'): ('研究区经历了长达一周的强降雨过程', '土壤墒情明显改善', '土壤墒情明显恶化'),
    ('天气', 'zh', 'casual'): ('这雨下了一整个礼拜都没停过', '地里的水管够了', '地里旱得裂口了'),
    ('天气', 'en', 'formal'): ('The region recorded persistent heavy rainfall over the past week',
                              'and soil moisture improved markedly', 'and soil moisture declined markedly'),
    ('天气', 'en', 'casual'): ('it rained hard all week long',
                              'so the yard is damp and soft', 'so the yard is dry and hard'),
    ('运动', 'zh', 'formal'): ('运动员完成了为期三个月的高原集训', '有氧耐力显著提升', '有氧耐力显著下降'),
    ('运动', 'zh', 'casual'): ('他天天早上五点起来跑步', '体力比以前好多了', '体力比以前差多了'),
    ('运动', 'en', 'formal'): ('The athletes completed a three-month high-altitude training camp',
                              'and their aerobic endurance improved', 'and their aerobic endurance declined'),
    ('运动', 'en', 'casual'): ('she trains at the gym every single day',
                              'so she got stronger and faster', 'so she got slower and weaker'),
    ('饮食', 'zh', 'formal'): ('受试者连续四周坚持低盐饮食方案', '血压水平稳步回落', '血压水平持续攀升'),
    ('饮食', 'zh', 'casual'): ('他这一个月顿顿都吃清水菜', '秤上的数字降了些', '秤上的数字涨了些'),
    ('饮食', 'en', 'formal'): ('The participants followed a low-salt diet for four consecutive weeks',
                              'and their blood pressure decreased', 'and their blood pressure increased'),
    ('饮食', 'en', 'casual'): ('he skipped sugary drinks for a month',
                              'so his jeans fit him again', 'so his jeans got way tighter'),
    ('学习', 'zh', 'formal'): ('实验组采用了间隔重复的记忆策略', '词汇测验成绩提高', '词汇测验成绩下滑'),
    ('学习', 'zh', 'casual'): ('她每天睡前背二十个单词', '听写基本都能过关', '听写回回都不及格'),
    ('学习', 'en', 'formal'): ('The experimental group used a spaced-repetition memory strategy',
                              'and their vocabulary scores rose', 'and their vocabulary scores fell'),
    ('学习', 'en', 'casual'): ('he crammed all night before the quiz',
                              'and he passed it easily', 'and he bombed it completely'),
    ('植物', 'zh', 'formal'): ('温室保持了恒定的光照与湿度', '幼苗长势整齐健壮', '幼苗成片枯萎发黄'),
    ('植物', 'zh', 'casual'): ('阳台那盆绿萝天天浇水施肥', '叶子绿得直发亮', '叶子黄得直掉渣'),
    ('植物', 'en', 'formal'): ('The greenhouse maintained constant light and humidity levels',
                              'and the seedlings grew vigorously', 'and the seedlings withered rapidly'),
    ('植物', 'en', 'casual'): ('she repotted the old ficus last spring',
                              'and it grew lots of new leaves', 'and it lost lots of old leaves'),
    ('市场', 'zh', 'formal'): ('节前冷链物流运力全面提升', '生鲜价格保持平稳', '生鲜价格大幅上涨'),
    ('市场', 'zh', 'casual'): ('早市新来了几家卖菜的摊位', '菜价比往常便宜些', '菜价比往常贵得多'),
    ('市场', 'en', 'formal'): ('Cold-chain capacity expanded ahead of the holiday season',
                              'and produce prices stayed stable', 'and produce prices rose sharply'),
    ('市场', 'en', 'casual'): ('the night market added a bunch of stalls',
                              'so the lines got even longer', 'so the lines got much shorter'),
    ('睡眠', 'zh', 'formal'): ('受试者连续两周保持规律作息', '白天精神状态饱满', '白天哈欠连绵不断'),
    ('睡眠', 'zh', 'casual'): ('他这两天十点就爬上床睡了', '早上起来特有劲', '早上起来特犯困'),
    ('睡眠', 'en', 'formal'): ('The participants kept a regular sleep schedule for two weeks',
                              'and their daytime alertness improved', 'and their daytime alertness dropped'),
    ('睡眠', 'en', 'casual'): ('she finally quit late-night scrolling',
                              'and woke up feeling refreshed', 'and woke up feeling groggy'),
    ('冷链', 'zh', 'formal'): ('新冷库投用后全程温控不断链', '疫苗批次全部合格', '疫苗批次大量报废'),
    ('冷链', 'zh', 'casual'): ('那家店进了台新的冰柜', '雪糕再也不化了', '雪糕化得软塌塌'),
    ('冷链', 'en', 'formal'): ('The new cold storage kept the temperature stable throughout',
                              'and the vaccine batches passed', 'and the vaccine batches failed'),
    ('冷链', 'en', 'casual'): ('the delivery van got a freezer upgrade',
                              'so the fish arrived still fresh', 'so the fish arrived half spoiled'),
}

def variant_texts(topic, lang, style, tail):
    """返回 [(prompt, prefix_str)] x 4 cuts —— 同字数移逗号变体; tail=B_con 或 B_contra"""
    A, Bcon, Bcot = MAT[(topic, lang, style)]
    B = tail
    out = []
    if lang == 'zh':
        n = len(A)
        for frac in CUT_FRACS:
            c = int(round(frac * n))
            assert 1 <= c < n, ('zh cut interior', topic, style, frac, c, n)
            pre = A[:c]
            prompt = A[:c] + '\uff0c' + A[c:] + B
            out.append((prompt, pre))
    else:
        ws = A.split(' ')
        n = len(ws)
        for frac in CUT_FRACS:
            w = int(round(frac * n))
            assert 1 <= w < n, ('en cut interior', topic, style, frac, w, n)
            pre = ' '.join(ws[:w])
            prompt = pre + ',' + ' ' + ' '.join(ws[w:]) + ' ' + B
            out.append((prompt, pre))
    return out

# ---- fail-fast 材料断言（任何 GPU 观测之前） ----
def material_asserts():
    notes = []
    for (tp, lg, st), (A, Bcon, Bcot) in MAT.items():
        if lg == 'zh':
            assert len(Bcon) == len(Bcot), ('zh tail length', tp, st, Bcon, Bcot)
        else:
            assert len(Bcon.split(' ')) == len(Bcot.split(' ')), ('en tail words', tp, st)
        vts = variant_texts(tp, lg, st, Bcon)
        lens = set(len(p) for p, _ in vts)
        assert len(lens) == 1, ('variant char invariance', tp, lg, st, lens)
        vts2 = variant_texts(tp, lg, st, Bcot)
        lens2 = set(len(p) for p, _ in vts2)
        # en 尾按预注册口径=词数相等, 字数可差 1-2（已知小混杂, 记入 result）
        assert len(lens2) == 1, ('contra invariance', tp, lg, st, lens2)
        if lg == 'en' and lens != lens2:
            print('note: en tail char length differs con=%d contra=%d (%s/%s)'
                  % (list(lens)[0], list(lens2)[0], tp, st))
        notes.append('%s/%s/%s ok(len=%d)' % (tp, lg, st, len(vts[0][0])))
    return notes

MAT_NOTES = material_asserts()

# ---------------- 行表构建 ----------------
def build_rows():
    rows = []
    topics_tr = TOPICS[:2] if SMOKE else TOPICS
    cuts = CUT_FRACS[:2] if SMOKE else CUT_FRACS
    for ti, tp in enumerate(topics_tr):
        for l, lg in enumerate(LANGS):
            for s, st in enumerate(STYLES):
                A_, Bcon_, Bcot_ = MAT[(tp, lg, st)]
                for g, gc in enumerate(LOGICS):
                    vts = variant_texts(tp, lg, st, Bcon_ if g == 0 else Bcot_)
                    keep = list(enumerate(vts))
                    if SMOKE:
                        keep = [kv for kv in keep if kv[0] < 2]
                    for ci, (prompt, pre) in keep:
                        rows.append(dict(ti=ti, topic=tp, l=l, lang=lg, s=s, style=st,
                                         g=g, logic=gc, ci=ci, cut=CUT_FRACS[ci],
                                         prompt=prompt, prefix=pre, ho=0))
    topics_ho = TOPICS_HO[:1] if SMOKE else TOPICS_HO
    base_ti = len(topics_tr)
    for ti, tp in enumerate(topics_ho):
        for l, lg in enumerate(LANGS):
            for s, st in enumerate(STYLES):
                A_, Bcon_, Bcot_ = MAT[(tp, lg, st)]
                for g, gc in enumerate(LOGICS):
                    vts = variant_texts(tp, lg, st, Bcon_ if g == 0 else Bcot_)
                    prompt, pre = vts[HO_CUT_IDX]
                    rows.append(dict(ti=base_ti + ti, topic=tp, l=l, lang=lg, s=s, style=st,
                                     g=g, logic=gc, ci=HO_CUT_IDX, cut=CUT_FRACS[HO_CUT_IDX],
                                     prompt=prompt, prefix=pre, ho=1))
    return rows

ROWS = build_rows()
NROWS = len(ROWS)
NTR = sum(1 for r in ROWS if r['ho'] == 0)
NHO = NROWS - NTR
EXP_TR = (2 * 2 * 2 * 2 * 2) if SMOKE else (6 * 2 * 2 * 2 * 4)
EXP_HO = (1 * 2 * 2 * 2) if SMOKE else (2 * 2 * 2 * 2)
assert NTR == EXP_TR and NHO == EXP_HO, ('panel size', NTR, NHO, EXP_TR, EXP_HO)
NBLK = (2 * 2 * 2) if SMOKE else (6 * 2 * 2)      # content 块数 = 主题x语言x风格
log('panel rows=%d (train %d + heldout %d), blocks=%d, smoke=%s, null_dirs=%d' %
    (NROWS, NTR, NHO, NBLK, SMOKE, NULL_DIRS))

# ---------------- 因素正交基（训练行, 预计算一次） ----------------
def orthonorm(M, drop_tol=1e-8):
    """SVD 正交化, 返回 (n, k) 列正交基"""
    U, s, Vt = np.linalg.svd(np.asarray(M, dtype=np.float64), full_matrices=False)
    keep = s > (s[0] * drop_tol if s.size and s[0] > 0 else 1e-12)
    return U[:, keep]

def orth_against(M, *qlist):
    R = np.asarray(M, dtype=np.float64).copy()
    for Q in qlist:
        R = R - Q @ (Q.T @ R)
    return R

def build_bases():
    tr = [i for i, r in enumerate(ROWS) if r['ho'] == 0]
    n = len(tr)
    vL = np.array([[1.0 if ROWS[i]['l'] == 0 else 0.0] for i in tr])
    vS = np.array([[1.0 if ROWS[i]['s'] == 0 else 0.0] for i in tr])
    # L, S 联合基（2 维, 相互正交）
    QLS = orthonorm(np.concatenate([vL, vS], 1))
    assert QLS.shape == (n, 2), ('QLS rank', QLS.shape)
    # content: NBLK 块指示, 剔除 L/S 张成后正交化
    blocks = np.zeros((n, NBLK))
    for j, i in enumerate(tr):
        blocks[j, ROWS[i]['ti'] * 4 + ROWS[i]['l'] * 2 + ROWS[i]['s']] = 1.0
    QC = orthonorm(orth_against(blocks, QLS))
    assert QC.shape == (n, NBLK - 2), ('QC rank', QC.shape)
    # G: contra 指示
    vG = np.array([[1.0 if ROWS[i]['g'] == 1 else 0.0] for i in tr])
    qG = orthonorm(orth_against(vG, QLS, QC))
    assert qG.shape == (n, 1), ('qG rank', qG.shape)
    return dict(tr=tr, QLS=QLS, QC=QC, qG=qG)

BASES = build_bases()

def shares_at(Hk, need_curve=False):
    """Hk: (NROWS, D) 某层隐状态; 返回训练行序贯 ANOVA 份额（和=1）"""
    tr = BASES['tr']
    X = Hk[tr].astype(np.float64)
    mu = X.mean(0)
    R0 = X - mu
    SS = float((R0 ** 2).sum()) + 1e-18
    e = {}
    resid = R0.copy()
    for nm, Q in [('L', BASES['QLS'][:, :1]), ('S', BASES['QLS'][:, 1:2]),
                  ('C', BASES['QC']), ('G', BASES['qG'])]:
        P = Q @ (Q.T @ resid)
        e[nm] = float((P ** 2).sum())
        resid = resid - P
    # D 为连续协变量: 投影基在调用处由 qD 给出（依赖 d 值, 见 build_qd）
    P = BASES['qD'] @ (BASES['qD'].T @ resid)
    e['D'] = float((P ** 2).sum())
    resid = resid - P
    e['R'] = float((resid ** 2).sum())
    sh = {k: v / SS for k, v in e.items()}
    tot = sum(sh.values())
    assert abs(tot - 1.0) < 1e-6, ('anova closure', tot)
    return sh

def build_qd(d_tr):
    """d 协变量基: 中心化后对 [L,S,C,G] 正交化"""
    v = np.asarray(d_tr, dtype=np.float64).reshape(-1, 1)
    v = v - v.mean()
    qD = orthonorm(orth_against(v, BASES['QLS'], BASES['QC'], BASES['qG']))
    assert qD.shape[1] == 1, ('qD rank', qD.shape)
    BASES['qD'] = qD
    return qD

def intervention(Hk, d_null_seed=0):
    """KOUT/KSTAR 几何干预: w_G flip 率 vs 随机方向 null"""
    tr = BASES['tr']
    X = Hk[tr].astype(np.float64)
    R0 = X - X.mean(0)
    g = np.array([ROWS[i]['g'] for i in tr])
    mcon = R0[g == 0].mean(0)
    mcot = R0[g == 1].mean(0)
    w = mcot - mcon
    w = w / (np.linalg.norm(w) + 1e-12)
    m = R0 @ w
    sep = float(m[g == 1].mean() - m[g == 0].mean())
    mc = m[g == 0] - m[g == 0].mean()
    mt = m[g == 1] - m[g == 1].mean()
    sd = float(np.concatenate([mc, mt]).std(ddof=1)) + 1e-12
    stat = sep / sd
    alpha = ALPHA_FRAC * sep
    flip_real = float(np.mean(m[g == 0] + alpha > 0.0))
    rng = np.random.RandomState(d_null_seed)
    stats, flips = [], []
    Dm = R0.shape[1]
    for _ in range(NULL_DIRS):
        u = rng.randn(Dm)
        u = u / np.linalg.norm(u)
        mu_ = R0 @ u
        sepu = float(mu_[g == 1].mean() - mu_[g == 0].mean())
        sdu = float(np.concatenate([mu_[g == 0] - mu_[g == 0].mean(),
                                    mu_[g == 1] - mu_[g == 1].mean()]).std(ddof=1)) + 1e-12
        stats.append(sepu / sdu)
        flips.append(float(np.mean(mu_[g == 0] + ALPHA_FRAC * sepu > 0.0)))
    stats = np.array(stats)
    p_stat = float(np.mean(stats >= stat))
    return dict(stat=stat, sep=sep, sd_within=sd, alpha=alpha,
                flip_real=flip_real, flip_null_mean=float(np.mean(flips)),
                flip_null_max=float(np.max(flips)),
                null_stat_mean=float(stats.mean()), null_stat_max=float(stats.max()),
                p_stat=p_stat, w_norm=float(np.linalg.norm(w)))

def heldout_eval(Hk, w, train_mean):
    ho = [i for i, r in enumerate(ROWS) if r['ho'] == 1]
    X = Hk[ho].astype(np.float64) - train_mean
    g = np.array([ROWS[i]['g'] for i in ho])
    m = X @ w
    acc = float(np.mean(np.sign(m) == np.where(g == 1, 1.0, -1.0)))
    # w_G 方向（D 空间 1 维子空间）对 held-out 残差能量份额（描述性;
    # 训练 qG 基在行空间, 对 held-out 行不可投, 故用 D 空间方向能量）
    g_share = float(((np.outer(m, w)) ** 2).sum() / ((X ** 2).sum() + 1e-18))
    return dict(acc=acc, g_share=g_share, n=len(ho),
                margin_mean_contra=float(m[g == 1].mean()),
                margin_mean_con=float(m[g == 0].mean()))

# ---------------- 模式 A/B/C: GPU 采集 + 分析 ----------------
if MODEL in ('qwen3-4b', 'qwen3-14b', 'glm4'):
    MDIR_MAP = {'qwen3-4b': 'qwen3-4b', 'qwen3-14b': 'Qwen3-14B', 'glm4': 'glm4-9b-chat-hf'}
    MDIR = os.path.join(ROOT, 'models', 'hf', MDIR_MAP[MODEL])
    cfg = json.load(open(os.path.join(MDIR, 'config.json'), encoding='utf-8'))
    NL = cfg['num_hidden_layers']
    HID = cfg['hidden_size']
    KSTAR = int(round(0.075 * NL))
    KOUT = NL - 1
    design = dict(model=MODEL, mdir=MDIR, phase='g1p4_mfd_multifactor_disentangle',
                  nl=NL, hidden=HID, kstar=KSTAR, readout=KOUT,
                  topics=TOPICS, topics_ho=TOPICS_HO, langs=LANGS, styles=STYLES,
                  logics=LOGICS, cut_fracs=list(CUT_FRACS), ho_cut=CUT_FRACS[HO_CUT_IDX],
                  n_rows=NROWS, n_train=NTR, n_heldout=NHO, n_blocks=NBLK,
                  anova_order='L -> S -> content(%d blocks) -> G -> D(continuous) -> resid' % NBLK,
                  fp_gate=FP_GATE, ho_gate=HO_GATE, alpha_frac=ALPHA_FRAC,
                  null_dirs=NULL_DIRS, d_def='n_tok(prompt) - n_tok(prefix)',
                  material={'%s|%s|%s' % k: v for k, v in MAT.items()}, material_asserts=MAT_NOTES,
                  smoke=SMOKE,
                  pre_reg='MEMO 2026-10-01 (redirect: original G2-P1 -> 3155); frozen before observation')
    exe_sha = freeze_design('g1p4_%s' % MODEL, design)
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

    # d 协变量（tokenizer 实测）
    for r in ROWS:
        n_full = len(tok(r['prompt'], add_special_tokens=False)['input_ids'])
        n_pre = len(tok(r['prefix'], add_special_tokens=False)['input_ids'])
        r['d'] = float(n_full - n_pre)
        assert r['d'] >= 1, ('d >= 1', r['prompt'], r['d'])
    # d 单调性（每组 4/2 变体内 d 应随 cut 递减）
    groups = {}
    for i, r in enumerate(ROWS):
        if r['ho'] == 0:
            groups.setdefault((r['topic'], r['lang'], r['style'], r['logic']), []).append(i)
    viol = 0
    for key, idxs in groups.items():
        ds = [ROWS[i]['d'] for i in idxs]
        if any(ds[j] <= ds[j + 1] for j in range(len(ds) - 1)):
            viol += 1
    viol_rate = viol / max(1, len(groups))
    log('d covariate: monotone violation rate=%.3f (%d/%d groups), d range=%.1f..%.1f' %
        (viol_rate, viol, len(groups),
         min(r['d'] for r in ROWS), max(r['d'] for r in ROWS)))
    assert viol_rate <= MONO_MAX_VIOL, ('D covariate degenerate', viol_rate)

    # 采集
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
                if (i + 1) % 32 == 0:
                    log('collect %d/%d' % (i + 1, NROWS))
        # 确定性锚: 3 行重前向位级比对
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
        det_note = 'bitwise' if det_ok else 'fp16 max abs diff %.3e (nonzero, tolerance pass)' % det_max
        assert det_max < 1e-3, ('determinism check fail', det_max)
        log('determinism recheck rows=%s -> %s' % (det_rows, det_note))
        np.savez_compressed(cache, H=H16,
                            d=np.array([r['d'] for r in ROWS], np.float32),
                            g=np.array([r['g'] for r in ROWS], np.int8),
                            l=np.array([r['l'] for r in ROWS], np.int8),
                            s=np.array([r['s'] for r in ROWS], np.int8),
                            ti=np.array([r['ti'] for r in ROWS], np.int8),
                            ci=np.array([r['ci'] for r in ROWS], np.int8),
                            ho=np.array([r['ho'] for r in ROWS], np.int8))
        log('collect saved %s' % os.path.basename(cache))
    npz_sha = hashlib.sha256(open(cache, 'rb').read()).hexdigest()[:8]
    del model
    torch.cuda.empty_cache()
    log('model released; npz sha8=%s' % npz_sha)

    # 分析（CPU）
    tr = BASES['tr']
    d_tr = [ROWS[i]['d'] for i in tr]
    build_qd(d_tr)
    def Hk(k):
        return H16[:, k, :].astype(np.float32)

    sh_kout = shares_at(Hk(KOUT))
    sh_kstar = shares_at(Hk(KSTAR))
    sh_emb = shares_at(Hk(0))
    fp_kout = [sh_kout[x] for x in ('L', 'S', 'C', 'G', 'D', 'R')]
    fp_kstar = [sh_kstar[x] for x in ('L', 'S', 'C', 'G', 'D', 'R')]
    log('shares KOUT: ' + ' '.join('%s=%.4f' % (k, v) for k, v in sh_kout.items()))
    log('shares KSTAR: ' + ' '.join('%s=%.4f' % (k, v) for k, v in sh_kstar.items()))

    # 全层 G/L/D 曲线
    curve_G, curve_L, curve_D = {}, {}, {}
    for k in range(NH):
        sh = shares_at(Hk(k))
        curve_G[str(k)] = sh['G']
        curve_L[str(k)] = sh['L']
        curve_D[str(k)] = sh['D']
    g_argmax = int(max(range(NH), key=lambda k: curve_G[str(k)]))

    # 几何干预
    iv_kout = intervention(Hk(KOUT))
    iv_kstar = intervention(Hk(KSTAR), d_null_seed=1)
    log('intervention KOUT: stat=%.3f p=%.4f flip=%.3f null=%.3f' %
        (iv_kout['stat'], iv_kout['p_stat'], iv_kout['flip_real'], iv_kout['flip_null_mean']))
    # w_G held-out 泛化
    Xtr_kout = Hk(KOUT)[tr].astype(np.float64)
    mcon = Xtr_kout[[j for j, i in enumerate(tr) if ROWS[i]['g'] == 0]].mean(0)
    mcot = Xtr_kout[[j for j, i in enumerate(tr) if ROWS[i]['g'] == 1]].mean(0)
    w_out = mcot - mcon
    w_out = w_out / (np.linalg.norm(w_out) + 1e-12)
    ho_out = heldout_eval(Hk(KOUT), w_out, Xtr_kout.mean(0))
    log('heldout KOUT: acc=%.3f g_share=%.4f' % (ho_out['acc'], ho_out['g_share']))

    verdict = ('g1p4_%s|G_%.4f|D_%.4f|resid_%.4f|flip_%.3f_null_%.3f_p_%.4f|hoacc_%.2f' %
               (MODEL, sh_kout['G'], sh_kout['D'], sh_kout['R'],
                iv_kout['flip_real'], iv_kout['flip_null_mean'], iv_kout['p_stat'],
                ho_out['acc']))
    result = dict(
        phase=3154, name=NAME, mode=MODEL,
        created=time.strftime('%Y-%m-%d %H:%M:%S'),
        runtime_s=round(time.time() - T0, 1),
        design_sha=exe_sha, smoke=SMOKE,
        n_rows=NROWS, n_train=NTR, n_heldout=NHO, n_blocks=NBLK,
        NL=NL, hidden=HID, kstar=KSTAR, readout=KOUT,
        npz=cache, npz_sha8=npz_sha, determinism_note=det_note,
        d_covariate=dict(violation_rate=viol_rate,
                         d_min=float(min(r['d'] for r in ROWS)),
                         d_max=float(max(r['d'] for r in ROWS)),
                         d_mean_by_cut={('%.2f' % CUT_FRACS[ci]): float(np.mean(
                             [r['d'] for r in ROWS if r['ho'] == 0 and r['ci'] == ci]))
                             for ci in range(len(CUT_FRACS))}),
        shares_kout=sh_kout, shares_kstar=sh_kstar, shares_emb=sh_emb,
        fp_kout=fp_kout, fp_kstar=fp_kstar,
        curve_G=curve_G, curve_L=curve_L, curve_D=curve_D,
        curve_G_argmax_layer=g_argmax,
        intervention_kout=iv_kout, intervention_kstar=iv_kstar,
        heldout_kout=ho_out,
        gates=dict(fp_gate=FP_GATE, ho_gate=HO_GATE, ho_pass=bool(ho_out['acc'] >= HO_GATE),
                   logic_axis_significant=bool(iv_kout['p_stat'] < 0.05)),
        grade='statistical',
        verdict=verdict)
    json.dump({'rows': [{k: r[k] for k in ('topic', 'lang', 'style', 'logic', 'cut',
                                           'prompt', 'prefix', 'd', 'ho')} for r in ROWS]},
              open(os.path.join(BASE, 'materials.json'), 'w', encoding='utf-8'),
              ensure_ascii=False, indent=1)
    seal_result(result, 'result.json')
    log('DONE mode=%s runtime %.1fs' % (MODEL, time.time() - T0))
    sys.exit(0)

# ---------------- 模式 D: summary —— 三模型指纹判决 ----------------
if MODEL == 'summary':
    P4B = os.path.join(RDIR, 'phase3154', NAME, 'qwen3-4b', 'result.json')
    P14B = os.path.join(RDIR, 'phase3154', NAME, 'qwen3-14b', 'result.json')
    PG4 = os.path.join(RDIR, 'phase3154', NAME, 'glm4', 'result.json')
    r4b = json.load(open(P4B, encoding='utf-8'))
    r14b = json.load(open(P14B, encoding='utf-8'))
    rg4 = json.load(open(PG4, encoding='utf-8'))
    design = dict(inputs={'qwen3-4b': dict(path=P4B, sha8=r4b.get('res_sha8'),
                                           seal=r4b.get('seal_sha8')),
                          'qwen3-14b': dict(path=P14B, sha8=r14b.get('res_sha8'),
                                            seal=r14b.get('seal_sha8')),
                          'glm4': dict(path=PG4, sha8=rg4.get('res_sha8'),
                                       seal=rg4.get('seal_sha8'))},
                  fp_gate=FP_GATE, ho_gate=HO_GATE,
                  pre_reg='MEMO 2026-10-01; fingerprint gate on KOUT 6-dim shares')
    exe_sha = freeze_design('g1p4_summary', design)
    fps = {'qwen3-4b': r4b['fp_kout'], 'qwen3-14b': r14b['fp_kout'], 'glm4': rg4['fp_kout']}
    fps_ks = {'qwen3-4b': r4b['fp_kstar'], 'qwen3-14b': r14b['fp_kstar'], 'glm4': rg4['fp_kstar']}
    pairs = {}
    keys = sorted(fps)
    for a, b in [('qwen3-4b', 'qwen3-14b'), ('qwen3-4b', 'glm4'), ('qwen3-14b', 'glm4')]:
        va, vb = np.array(fps[a]), np.array(fps[b])
        r = float(np.corrcoef(va, vb)[0, 1])
        rk = float(np.corrcoef(np.array(fps_ks[a]), np.array(fps_ks[b]))[0, 1])
        pairs['%s_vs_%s' % (a, b)] = dict(pearson_kout=r, pearson_kstar=rk,
                                          pass_gate=bool(r >= FP_GATE))
        log('fp %s vs %s: kout r=%.4f kstar r=%.4f' % (a, b, r, rk))
    fpmin = min(p['pearson_kout'] for p in pairs.values())
    all_pass = all(p['pass_gate'] for p in pairs.values())
    ho_all = all(x['heldout_kout']['acc'] >= HO_GATE for x in (r4b, r14b, rg4))
    iv_all = all(x['gates']['logic_axis_significant'] for x in (r4b, r14b, rg4))
    if all_pass:
        verdict = 'g1p4_fingerprint_consistent|fpmin_%.3f|ho_pass_%d/3|logic_sig_%d/3' % (
            fpmin, sum(1 for x in (r4b, r14b, rg4) if x['heldout_kout']['acc'] >= HO_GATE),
            sum(1 for x in (r4b, r14b, rg4) if x['gates']['logic_axis_significant']))
    else:
        bad = [k for k, p in pairs.items() if not p['pass_gate']]
        verdict = ('g1p4_fingerprint_inconsistent_material_method_descriptive|fpmin_%.3f|bad_pairs_%s' %
                   (fpmin, '+'.join(bad)))
    per_model = {}
    for mn, rr in [('qwen3-4b', r4b), ('qwen3-14b', r14b), ('glm4', rg4)]:
        per_model[mn] = dict(
            NL=rr['NL'], kstar=rr['kstar'], readout=rr['readout'],
            shares_kout=rr['shares_kout'], shares_kstar=rr['shares_kstar'],
            curve_G_argmax=rr['curve_G_argmax_layer'],
            iv_kout=dict(stat=rr['intervention_kout']['stat'],
                         p=rr['intervention_kout']['p_stat'],
                         flip=rr['intervention_kout']['flip_real'],
                         flip_null=rr['intervention_kout']['flip_null_mean']),
            heldout_acc=rr['heldout_kout']['acc'],
            d_mean_by_cut=rr['d_covariate']['d_mean_by_cut'],
            verdict=rr['verdict'])
    result = dict(
        phase=3154, name=NAME, mode='summary',
        created=time.strftime('%Y-%m-%d %H:%M:%S'),
        runtime_s=round(time.time() - T0, 1),
        design_sha=exe_sha, smoke=SMOKE,
        fp_pairs=pairs, fp_min_kout=fpmin, fingerprint_consistent=bool(all_pass),
        ho_all_pass=bool(ho_all), logic_axis_sig_all=bool(iv_all),
        per_model=per_model,
        shares_mean_kout={k: float(np.mean([fps[m][i] for m in keys]))
                          for i, k in enumerate(('L', 'S', 'C', 'G', 'D', 'R'))},
        inputs_used=design['inputs'],
        grade='statistical',
        verdict=verdict)
    seal_result(result, 'result_summary.json')
    log('DONE mode=summary runtime %.1fs' % (time.time() - T0))
    sys.exit(0)

log('unknown mode %s' % MODEL)
sys.exit(2)
