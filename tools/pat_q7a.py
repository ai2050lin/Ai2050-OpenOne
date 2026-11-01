# -*- coding: utf-8 -*-
"""Q7-A：句间关系是否因子化（顺承 / 转折 / 因果）。

预注册：research/gpt5/docs/PAT_Q7A_PREREG_v1.md（§1–§5 冻结，§6 跑前修订）

三因子：REL(3) × CONT(18) × SURF(3) = 162 句。
主判全部预测型（D24）：留出内容的分类准确率，不是剖面份额。

J0 对齐自检    P_AE 上准确率必须恰等于 0.3333（同内容同表面三点输入逐位相同）
J1 稳定        P_END 留出内容 3 分类，6 折按内容，l* 由内层 5 折选
J2 表面解耦    S1+S2(有连接词) -> S3(无连接词) 迁移；辅 S3->S1S2 与 S1->S2 最小对
J3 复用        W 的秩-k 截断曲线（k=1,2,3）
G0 刺激闸      隐式句对的元语言三选一，跑前先验材料

用法
----
  SMOKE=1 .venv/Scripts/python.exe -X utf8 tools/pat_q7a.py
  .venv/Scripts/python.exe -X utf8 tools/pat_q7a.py
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
for _p in (str(ROOT), str(ROOT / 'tools')):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import provenance

SMOKE = os.environ.get('SMOKE', '') == '1'

MODEL_PATH = 'D:/develop/model/Qwen/Qwen3-4B'
SEED = 20261101

SYS = '你是一个阅读助手。'

# ---------------- 材料（预注册 §1，冻结） ----------------
# 每个内容：A 为第一句；B 的三种写法共用两个主题词（kw），只改谓语立场。
CONTENTS = [
    dict(a='昨天那场评审会开了三个小时，好几个模块都没过',
         seq='今天上午又开了一次评审会，把剩下的模块过完',
         con='今天上午的评审会上，那几个模块一份没改就放行了',
         cau='今天上午的评审会只能重开，那几个模块全部推倒重做',
         kw=['评审会', '模块']),
    dict(a='台风昨天半夜扫过，机场的航班全停了',
         seq='今天早上机场重新排班，把停掉的航班挨个放行',
         con='今天早上机场的航班全按原计划起飞了',
         cau='今天早上的航班大面积延误，好多人还堵在机场',
         kw=['航班', '机场']),
    dict(a='火锅的锅底刚端上来就糊了',
         seq='服务员换了新的锅底，火锅重新烧开',
         con='服务员说锅底没问题，火锅不用换',
         cau='锅底只能倒掉，我们换了一家店重新点火锅',
         kw=['锅底', '火锅']),
    dict(a='手机用了两年，电池一天要充电三次',
         seq='我换了新电池，又把充电的习惯改了一遍',
         con='我换了新电池，一天要充电的次数一点没少',
         cau='电池换不了，充电宝成了每天的必需品',
         kw=['电池', '充电']),
    dict(a='论文的第二组实验跑出来，数据全偏了',
         seq='我把数据重新导了一遍，实验照着原计划往下跑',
         con='数据我一个字没改，实验结论照原样写进了论文',
         cau='整个实验得推倒重做，数据也要重新采',
         kw=['实验', '数据']),
    dict(a='早高峰的地铁在换乘站挤得根本上不去',
         seq='地铁来了三趟，我跟着人流换乘上了后面那班',
         con='地铁第一趟我就上去了，换乘一分钟都没等',
         cau='地铁换乘耽误了半小时，我迟到了，全勤也没了',
         kw=['地铁', '换乘']),
    dict(a='阳台上的兰花浇水浇多了，根开始发黑',
         seq='我把兰花搬到通风处，浇水的次数也减了一半',
         con='兰花搬到通风处放了两周，浇水的次数一点没减，根一点没变好',
         cau='兰花最后烂掉了，我只能重新买一盆，浇水也不敢再乱来',
         kw=['兰花', '浇水']),
    dict(a='昨晚的比赛我们队上半场就丢了两个球，教练把前锋叫到场边说了很久',
         seq='下半场教练换上新前锋，我们连追两球',
         con='下半场教练换上新前锋，比分一点没变',
         cau='下半场教练把前锋全换下，队员心态崩了，最后输了四个',
         kw=['前锋', '教练']),
    dict(a='复诊的时候医生说指标全正常，药量可以停了',
         seq='我按医生说的先减半药量，等下次复诊再看',
         con='下次复诊前我没停药，药量照原样一点没减',
         cau='我把药量停了，第三天指标就反弹，只能再去复诊',
         kw=['复诊', '药量']),
    dict(a='装修的预算报到一半就超了',
         seq='装修的材料单我重新核了一遍，接着跟工长逐项对预算',
         con='装修的材料单我核了三遍，预算一分都没降下来',
         cau='装修的预算砍不动，只能少装两个柜子',
         kw=['装修', '预算']),
    dict(a='到酒店办入住时，我发现行李在托运时被摔裂了',
         seq='酒店前台帮我叫了维修，行李先寄存在大堂',
         con='酒店前台说这不归他们管，行李只能自己扛上楼',
         cau='行李裂得太厉害，酒店帮我联系了快递，直接寄回家',
         kw=['酒店', '行李']),
    dict(a='新版本上线之后，线上一直报错',
         seq='我把版本先回滚了，接着逐条看报错的日志',
         con='版本我回滚了，报错一点没少，日志里全是同一条',
         cau='报错停不下来，版本只能全部重发，服务停了四十分钟',
         kw=['版本', '报错']),
    dict(a='快递在驿站放了三天，包裹的外箱压扁了',
         seq='我去驿站把包裹取回来，接着拍照联系快递重发',
         con='我在驿站拆开看了，包裹里的东西一点没坏',
         cau='包裹里的杯子碎了，驿站说只能找快递理赔',
         kw=['包裹', '驿站']),
    dict(a='健身房买了二十节私教课，器械区一直排不上队',
         seq='我跟私教商量，把器械的时间挪到早上',
         con='我跟私教说了三次，器械一次都没排上',
         cau='私教说器械排不上没办法，剩下的课只能退',
         kw=['私教', '器械']),
    dict(a='从图书馆借的书拖了两个月，已经逾期了',
         seq='我今天去图书馆还书，接着把逾期的罚款交了',
         con='我今天去图书馆还书，系统里查不到逾期记录',
         cau='逾期的罚款加滞纳金，比书价还贵，图书馆还停了我的借书权限',
         kw=['图书馆', '逾期']),
    dict(a='演唱会的门票开售三分钟就抢光了，好座位一张不剩',
         seq='我刷了半小时，接着在二手平台上买到了门票，座位在内场',
         con='我刷了半小时，门票和座位一张都没刷出来',
         cau='门票没抢到，只能等下一场，座位也从内场换成了看台',
         kw=['门票', '座位']),
    dict(a='银行的贷款审批卡在面签，利率也涨了',
         seq='客户经理让我补贷款材料，接着把利率和面签重新约了一遍',
         con='客户经理说贷款材料不用补，利率也不变，面签照原时间进行',
         cau='贷款拖到下个月，能拿到的利率又高了一档',
         kw=['贷款', '利率']),
    dict(a='猫打完疫苗当天晚上，体温升到三十九度',
         seq='疫苗打完先观察，我接着每小时量一次体温',
         con='疫苗打完我观察了一夜，体温一点没降',
         cau='疫苗的反应一直不退，体温也降不下来，第二天只能抱回诊室',
         kw=['疫苗', '体温']),
]

RELS = ['seq', 'con', 'cau']
REL_CN = {'seq': '顺承', 'con': '转折', 'cau': '因果'}
CONN = {'seq': ['然后', '接着', '随后'],
        'con': ['但是', '可是', '不过'],
        'cau': ['所以', '因此', '于是']}
SURFS = ['S1', 'S2', 'S3']

# ---------------- 阈值（预注册 §2，冻结） ----------------
FLOOR = 1.0 / 3.0
TH_G0_VALID = 0.50
TH_G0_GRAY = 0.40
TH_J1_PASS = 0.60
TH_J1_FRAME = 0.45
TH_J2_PASS = 0.55
TH_J3_TOL = 0.05
ALPHA_REL = [0.01, 0.1, 1.0]
N_INNER = 5
N_OUTER = 6
N_NULL = 10


# ---------------- 构造刺激 ----------------
def build_stimuli():
    """返回句子表；同时做预注册 §1.2 的两条断言。"""
    rows = []
    for ci, c in enumerate(CONTENTS):
        # 断言 1：两个主题词都出现在 A 与全部 3 个 B 中
        for k in c['kw']:
            if k not in c['a']:
                raise AssertionError('内容%d 主题词 %s 不在 A' % (ci, k))
            for r in RELS:
                if k not in c[r]:
                    raise AssertionError('内容%d 主题词 %s 不在 B[%s]' % (ci, k, r))
        for r in RELS:
            conn = CONN[r][ci % 3]
            a, b = c['a'], c[r]
            for si, sf in enumerate(SURFS):
                if sf == 'S1':
                    text = a + '，' + conn + '，' + b + '。'
                    cs = len(a) + 1
                    ce = cs + len(conn)
                elif sf == 'S2':
                    text = a + '。' + conn + '，' + b + '。'
                    cs = len(a) + 1
                    ce = cs + len(conn)
                else:
                    text = a + '。' + b + '。'
                    cs = ce = -1
                rows.append(dict(ci=ci, rel=r, surf=sf, conn=conn, text=text,
                                 c_a_end=len(a) - 1, c_conn=cs, c_conn_end=ce,
                                 c_body_end=len(text) - 1))
    # 断言 2：主题词跨内容不相交
    for ci, c in enumerate(CONTENTS):
        for cj, d in enumerate(CONTENTS):
            if ci == cj:
                continue
            for k in c['kw']:
                for r in RELS:
                    blob = d['a'] + d[r]
                    if k in blob:
                        raise AssertionError('主题词 %s（内容%d）出现在内容%d' % (k, ci, cj))
        # 自身：任一主题词不得出现在其它内容的句子里已由上面覆盖
    return rows


def render(tok, user_text):
    return tok.apply_chat_template(
        [{'role': 'system', 'content': SYS}, {'role': 'user', 'content': user_text}],
        add_generation_prompt=True, tokenize=False, enable_thinking=False)


def tok_at(offs, c):
    for i, (x, y) in enumerate(offs):
        if x <= c < y:
            return i
    return len(offs) - 1


# ---------------- 探针 ----------------
def _std_fit(Xtr, Xte):
    m = Xtr.mean(0)
    s = Xtr.std(0)
    floor = 0.10 * float(s.mean()) + 1e-8
    s = np.maximum(s, floor)
    return (Xtr - m) / s, (Xte - m) / s


def _dual(Ztr, Ytr, alphas):
    K = (Ztr @ Ztr.T).astype(np.float64)
    K = 0.5 * (K + K.T)
    w, V = np.linalg.eigh(K)
    w = np.maximum(w, 1e-10)
    VY = V.T @ Ytr
    out = {}
    for a in alphas:
        A = V @ (VY / (w[:, None] + a))
        out[a] = A
    return out


# ---------------- 主流程 ----------------
def main() -> int:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    t0 = time.time()
    rng = np.random.default_rng(SEED)
    dev = 'cuda'

    rows = build_stimuli()
    n_sent = len(rows)
    n_cont = len(CONTENTS)
    y = np.array([RELS.index(r['rel']) for r in rows])
    ci_arr = np.array([r['ci'] for r in rows])
    sf_arr = np.array([r['surf'] for r in rows])
    print('[材料] %d 句 / %d 内容 / %d 关系 / %d 表面（断言全过）'
          % (n_sent, n_cont, len(RELS), len(SURFS)))

    tok = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH, dtype=torch.bfloat16, attn_implementation='eager').to(dev).eval()
    L = model.config.num_hidden_layers
    H = model.config.hidden_size
    NL = L + 1
    print('[模型] %d 层 / %d 维' % (L, H))

    # ---------- 前向：取三个位置的残差流 ----------
    pos_names = ['P_END', 'P_CONN', 'P_AE']
    feats = {p: np.zeros((NL, n_sent, H), dtype=np.float32) for p in pos_names}
    lens = np.zeros(n_sent, dtype=int)
    idx_pos = np.zeros((n_sent, len(pos_names)), dtype=int)
    bad = 0
    with torch.no_grad():
        for i, r in enumerate(rows):
            full = render(tok, r['text'])
            base = full.find(r['text'])
            if base < 0:
                raise RuntimeError('渲染文本中找不到用户内容')
            enc = tok(full, return_offsets_mapping=True, add_special_tokens=False)
            ids = enc['input_ids']
            offs = enc['offset_mapping']
            p_end = tok_at(offs, base + r['c_body_end'])
            p_ae = tok_at(offs, base + r['c_a_end'])
            if r['surf'] == 'S3':
                p_conn = p_end
            else:
                p_conn = tok_at(offs, base + r['c_conn'])
                if tok_at(offs, base + r['c_conn_end'] - 1) != p_conn:
                    bad += 1
            idx_pos[i] = (p_end, p_conn, p_ae)
            lens[i] = len(ids)
            t = torch.tensor([ids], device=dev)
            o = model(t, output_hidden_states=True)
            hs = torch.stack([h[0] for h in o.hidden_states], 0).float().cpu().numpy()
            feats['P_END'][:, i, :] = hs[:, p_end, :]
            feats['P_AE'][:, i, :] = hs[:, p_ae, :]
            feats['P_CONN'][:, i, :] = hs[:, p_conn, :]
    print('[前向] %d 句 完成，连接词跨 token 计数=%d，句长中位=%d'
          % (n_sent, bad, int(np.median(lens))))

    # ---------- G0 刺激有效性闸 ----------
    lab_tok = []
    for r in RELS:
        tid = tok(REL_CN[r], add_special_tokens=False)['input_ids']
        lab_tok.append(tid[0])
    g0_pred, g0_ok = [], 0
    n_g0 = 0
    with torch.no_grad():
        for cni in range(n_cont):
            c = CONTENTS[cni]
            for ri, r in enumerate(RELS):
                q = '%s。%s。这两句之间是什么关系？只回答一个词：顺承、转折、因果。答：' % (c['a'], c[r])
                full = render(tok, q)
                ids = tok(full, add_special_tokens=False)['input_ids']
                t = torch.tensor([ids], device=dev)
                o = model(t)
                lg = o.logits[0, -1, :].float().cpu().numpy()
                sc = [lg[x] for x in lab_tok]
                pr = int(np.argmax(sc))
                g0_pred.append(pr)
                g0_ok += int(pr == ri)
                n_g0 += 1
    g0_acc = g0_ok / n_g0
    g0_hist = [int(np.sum(np.array(g0_pred) == k)) for k in range(3)]
    print('[G0a] 隐式句对元语言三选一 acc=%.4f（地板 0.3333，n=%d）hist=%s'
          % (g0_acc, n_g0, g0_hist))

    # ---------- G0b 连接词契合度（B 条件似然，去偏） ----------
    # 口径：给 (A, 候选连接词) 求 B 的 token 条件似然；逐连接词减全局均值去偏。
    cand = ['然后', '但是', '所以']
    Sfit = np.zeros((n_g0, 3))
    ii = 0
    with torch.no_grad():
        for cni in range(n_cont):
            c = CONTENTS[cni]
            for r in RELS:
                pre_list = [c['a'] + '。' + cd + '，' for cd in cand]
                for j, pre in enumerate(pre_list):
                    txt = pre + c[r] + '。'
                    full = render(tok, txt)
                    base = full.find(txt)
                    enc = tok(full, return_offsets_mapping=True, add_special_tokens=False)
                    ids, offs = enc['input_ids'], enc['offset_mapping']
                    p0 = tok_at(offs, base + len(pre))
                    p1 = tok_at(offs, base + len(pre + c[r]))
                    t = torch.tensor([ids], device=dev)
                    lg = model(t).logits[0].float()
                    lsm = torch.log_softmax(lg, -1)
                    Sfit[ii, j] = float(sum(float(lsm[k, ids[k + 1]]) for k in range(p0, p1)))
                ii += 1
    g0b_bias = Sfit.mean(0)
    truth = np.tile(np.arange(3), n_cont)          # 外层按内容、内层按关系
    g0b_raw = float((Sfit.argmax(1) == truth).mean())
    g0b_deb = float(((Sfit - g0b_bias).argmax(1) == truth).mean())
    conf_b = np.zeros((3, 3), int)
    for t_ in range(n_g0):
        conf_b[truth[t_], int((Sfit - g0b_bias)[t_].argmax())] += 1
    g0b_per_rel = [round(float(conf_b[j, j] / conf_b[j].sum()), 3) for j in range(3)]
    print('[G0b] 连接词契合度 原始=%.4f 去偏=%.4f（偏差 logp=%s）每关系=%s'
          % (g0b_raw, g0b_deb, np.round(g0b_bias, 2).tolist(), g0b_per_rel))

    if SMOKE:
        print('[SMOKE] G0 已跑；管线形状 前向/索引/断言 全过。耗时 %.0fs' % (time.time() - t0))
        return 0

    # ---------- 若材料整体崩塌，直接停（§6.2：只保留「粗失败」闸） ----------
    if g0b_deb < 0.30:
        print('[中止] G0b 去偏 < 0.30 ⇒ STIMULI_INVALID，不进入 J1–J3')
        payload = dict(stage='G0_only', g0_acc=round(g0_acc, 4), g0_hist=g0_hist,
                       g0b_raw=round(g0b_raw, 4), g0b_deb=round(g0b_deb, 4),
                       g0b_per_rel=g0b_per_rel, n_sent=n_sent, verdict='STIMULI_INVALID',
                       elapsed_s=round(time.time() - t0, 1))
        provenance.write_result('pat_q7a', payload, model_path=MODEL_PATH,
                                seed=SEED, tool='tools/pat_q7a.py')
        return 0

    # ---------- 训练/测试划分（6 折按内容，冻结） ----------
    folds = []
    for f in range(N_OUTER):
        te = [i for i in range(n_cont) if i % N_OUTER == f]
        tr = [i for i in range(n_cont) if i % N_OUTER != f]
        folds.append((np.array(tr), np.array(te)))

    def rows_of(contents):
        return np.where(np.isin(ci_arr, contents))[0]

    # ---------- alpha 选择（层 17、P_END，内层 5 折按内容，一次冻结） ----------
    l_ref = min(17, L)
    a_scores = {a: [] for a in ALPHA_REL}
    inner_splits = []
    for f in range(N_INNER):
        te = [i for i in range(n_cont) if i % N_INNER == f]
        tr = [i for i in range(n_cont) if i % N_INNER != f]
        inner_splits.append((np.array(tr), np.array(te)))
    X = feats['P_END'][l_ref]
    for tr_c, te_c in inner_splits:
        rtr, rte = rows_of(tr_c), rows_of(te_c)
        Ztr, Zte = _std_fit(X[rtr], X[rte])
        Y = np.zeros((len(rtr), 3))
        Y[np.arange(len(rtr)), y[rtr]] = 1.0
        As = _dual(Ztr, Y, [a * len(rtr) for a in ALPHA_REL])
        for a_rel in ALPHA_REL:
            A = As[a_rel * len(rtr)]
            S = Zte @ (Ztr.T @ A)
            a_scores[a_rel].append(float((S.argmax(1) == y[rte]).mean()))
    alpha_rel = max(ALPHA_REL, key=lambda a: float(np.mean(a_scores[a])))
    print('[alpha] 层%d 内层候选 %s -> 选 %.3f（内层 acc=%s）'
          % (l_ref, ALPHA_REL, alpha_rel,
             {k: round(float(np.mean(v)), 4) for k, v in a_scores.items()}))

    # ---------- J1：逐层留出（外层 6 折），l* 由内层 5 折选 ----------
    prof = {p: np.zeros(NL) for p in pos_names}
    pred_store = {p: np.full(n_sent, -1, dtype=int) for p in pos_names}
    lstar = np.zeros(N_OUTER, dtype=int)
    fold_scores = {}

    for fi, (tr_c, te_c) in enumerate(folds):
        rtr_all, rte_all = rows_of(tr_c), rows_of(te_c)
        # 内层选层：内层 5 折 = 折内再按内容切（外层训练内容重新编号）
        sub = []
        for g in range(N_INNER):
            te2 = [c for j, c in enumerate(tr_c) if j % N_INNER == g]
            tr2 = [c for j, c in enumerate(tr_c) if j % N_INNER != g]
            sub.append((np.array(tr2), np.array(te2)))
        acc_by_layer = np.zeros(NL)
        for li in range(NL):
            accs = []
            Xl = feats['P_END'][li]
            for tr2, te2 in sub:
                r2, e2 = rows_of(tr2), rows_of(te2)
                Ztr, Zte = _std_fit(Xl[r2], Xl[e2])
                Y = np.zeros((len(r2), 3))
                Y[np.arange(len(r2)), y[r2]] = 1.0
                A = _dual(Ztr, Y, [alpha_rel * len(r2)])[alpha_rel * len(r2)]
                S = Zte @ (Ztr.T @ A)
                accs.append(float((S.argmax(1) == y[e2]).mean()))
            acc_by_layer[li] = float(np.mean(accs))
        ls = int(np.argmax(acc_by_layer))
        lstar[fi] = ls
        fold_scores[fi] = acc_by_layer
        # 外层评测（该折 l* 与 alpha）
        for p in pos_names:
            Xl = feats[p][ls]
            Ztr, Zte = _std_fit(Xl[rtr_all], Xl[rte_all])
            Y = np.zeros((len(rtr_all), 3))
            Y[np.arange(len(rtr_all)), y[rtr_all]] = 1.0
            A = _dual(Ztr, Y, [alpha_rel * len(rtr_all)])[alpha_rel * len(rtr_all)]
            S = Zte @ (Ztr.T @ A)
            pr = S.argmax(1)
            pred_store[p][rte_all] = pr

    for p in pos_names:
        prof[p] = np.array([0.0] * NL)
    # 逐层剖面（描述性，固定 alpha，外层 6 折合并）
    for li in range(NL):
        ok = {p: 0 for p in pos_names}
        for fi, (tr_c, te_c) in enumerate(folds):
            rtr_all, rte_all = rows_of(tr_c), rows_of(te_c)
            for p in pos_names:
                Xl = feats[p][li]
                Ztr, Zte = _std_fit(Xl[rtr_all], Xl[rte_all])
                Y = np.zeros((len(rtr_all), 3))
                Y[np.arange(len(rtr_all)), y[rtr_all]] = 1.0
                A = _dual(Ztr, Y, [alpha_rel * len(rtr_all)])[alpha_rel * len(rtr_all)]
                S = Zte @ (Ztr.T @ A)
                ok[p] += int((S.argmax(1) == y[rte_all]).sum())
        for p in pos_names:
            prof[p][li] = ok[p] / n_sent

    acc_J1 = float((pred_store['P_END'] == y).mean())
    pred_lstar = pred_store['P_END']

    def boot_acc(correct, groups, n_boot=2000):
        uniq = np.unique(groups)
        vals = []
        for _ in range(n_boot):
            pick = rng.choice(uniq, size=len(uniq), replace=True)
            m = np.concatenate([np.where(groups == g)[0] for g in pick])
            vals.append(float(correct[m].mean()))
        return np.array(vals)

    corr = (pred_store['P_END'] == y).astype(float)
    bs = boot_acc(corr, ci_arr)
    ci_j1 = (float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5)))
    if not (ci_j1[0] - 1e-9 <= acc_J1 <= ci_j1[1] + 1e-9):
        raise AssertionError('D26 违约：CI %s 未包住点估计 %.4f' % (ci_j1, acc_J1))
    print('[J1] P_END acc=%.4f  CI=[%.4f, %.4f]  离地板 %+.4f  l*=%s'
          % (acc_J1, ci_j1[0], ci_j1[1], acc_J1 - FLOOR, lstar.tolist()))

    # ---------- J0 对齐自检 ----------
    j0 = {}
    for sf in SURFS:
        m = np.where(sf_arr == sf)[0]
        j0[sf] = float((pred_store['P_AE'][m] == y[m]).mean())
    j0_max_dev = max(abs(v - FLOOR) for v in j0.values())
    print('[J0] P_AE 各表面 acc=%s  最大偏离=%.6f'
          % ({k: round(v, 6) for k, v in j0.items()}, j0_max_dev))

    # ---------- J2 跨表面迁移 ----------
    def transfer(train_surfs, test_surfs, ls_mode=True):
        correct = np.zeros(n_sent, dtype=bool)
        hit = np.zeros(n_sent, dtype=bool)
        preds = np.full(n_sent, -1, dtype=int)
        for fi, (tr_c, te_c) in enumerate(folds):
            ls = int(lstar[fi]) if ls_mode else l_ref
            Xl = feats['P_END'][ls]
            rtr = np.intersect1d(rows_of(tr_c), np.where(np.isin(sf_arr, train_surfs))[0])
            rte = np.intersect1d(rows_of(te_c), np.where(np.isin(sf_arr, test_surfs))[0])
            Ztr, Zte = _std_fit(Xl[rtr], Xl[rte])
            Y = np.zeros((len(rtr), 3))
            Y[np.arange(len(rtr)), y[rtr]] = 1.0
            A = _dual(Ztr, Y, [alpha_rel * len(rtr)])[alpha_rel * len(rtr)]
            S = Zte @ (Ztr.T @ A)
            preds[rte] = S.argmax(1)
            correct[rte] = S.argmax(1) == y[rte]
            hit[rte] = True
        return float(correct[hit].mean()), preds, hit

    j2_main, j2_pred, j2_hit = transfer(['S1', 'S2'], ['S3'])
    j2_rev, j2r_pred, j2r_hit = transfer(['S3'], ['S1', 'S2'])
    j2_punct, j2p_pred, j2p_hit = transfer(['S1'], ['S2'])
    j2_corr = (j2_pred == y).astype(float)[j2_hit]
    bs2 = boot_acc(j2_corr, ci_arr[j2_hit])
    ci2 = (float(np.percentile(bs2, 2.5)), float(np.percentile(bs2, 97.5)))
    if not (ci2[0] - 1e-9 <= j2_main <= ci2[1] + 1e-9):
        raise AssertionError('D26 违约：J2 CI 未包住点估计')
    print('[J2] S1S2->S3 %.4f CI=[%.4f, %.4f] | S3->S1S2 %.4f | S1->S2 %.4f'
          % (j2_main, ci2[0], ci2[1], j2_rev, j2_punct))

    acc_conn = float((pred_store['P_CONN'][sf_arr != 'S3'] == y[sf_arr != 'S3']).mean())
    print('[J2辅] P_CONN acc=%.4f（仅显式表面，连接词位置本身）' % acc_conn)

    # ---------- J3 秩截断曲线 ----------
    j3 = {}
    j3_pred = {}
    for k in (1, 2, 3):
        correct = np.zeros(n_sent, dtype=bool)
        hit = np.zeros(n_sent, dtype=bool)
        pk = np.full(n_sent, -1, dtype=int)
        for fi, (tr_c, te_c) in enumerate(folds):
            ls = int(lstar[fi])
            Xl = feats['P_END'][ls]
            rtr, rte = rows_of(tr_c), rows_of(te_c)
            Ztr, Zte = _std_fit(Xl[rtr], Xl[rte])
            Y = np.zeros((len(rtr), 3))
            Y[np.arange(len(rtr)), y[rtr]] = 1.0
            A = _dual(Ztr, Y, [alpha_rel * len(rtr)])[alpha_rel * len(rtr)]
            W = Ztr.T @ A                      # d x 3
            U, s, Vt = np.linalg.svd(W, full_matrices=False)
            Wk = (U[:, :k] * s[:k]) @ Vt[:k, :]
            pr = (Zte @ Wk).argmax(1)
            pk[rte] = pr
            correct[rte] = pr == y[rte]
            hit[rte] = True
        j3[k] = float(correct[hit].mean())
        j3_pred[k] = pk
    print('[J3] rank1=%.4f rank2=%.4f rank3=%.4f' % (j3[1], j3[2], j3[3]))

    # ---------- 零分布：内容内标签循环错位 ----------
    null_accs = []
    for t in range(N_NULL):
        yp = y.copy()
        for cni in range(n_cont):
            # 每内容一个独立置换：句子的「真标签」被随机改派，跨内容不一致
            perm = rng.permutation(3)
            for sf in SURFS:
                mm = np.where((ci_arr == cni) & (sf_arr == sf))[0]
                yp[mm] = perm
        corr_n = np.zeros(n_sent)
        for fi, (tr_c, te_c) in enumerate(folds):
            ls = int(lstar[fi])
            Xl = feats['P_END'][ls]
            rtr, rte = rows_of(tr_c), rows_of(te_c)
            Ztr, Zte = _std_fit(Xl[rtr], Xl[rte])
            Yn = np.zeros((len(rtr), 3))
            Yn[np.arange(len(rtr)), yp[rtr]] = 1.0
            A = _dual(Ztr, Yn, [alpha_rel * len(rtr)])[alpha_rel * len(rtr)]
            S = Zte @ (Ztr.T @ A)
            corr_n[rte] = (S.argmax(1) == yp[rte]).astype(float)
        null_accs.append(float(corr_n.mean()))
    null_mean = float(np.mean(null_accs))
    null_p95 = float(np.percentile(null_accs, 95))
    print('[零分布] 10 次内容内标签错位：mean=%.4f p95=%.4f' % (null_mean, null_p95))

    # ---------- MDE（内容簇口径，模拟） ----------
    def mde(correct, groups, n_sim=600, force_n=None):
        uniq = np.unique(groups)
        if force_n is None:
            n_per = np.array([int((groups == g).sum()) for g in uniq], dtype=float)
        else:
            n_per = np.full(len(uniq), float(force_n))
        rr = np.random.default_rng(SEED + 7)
        for delta in np.arange(0.05, 0.66, 0.025):
            true_a = FLOOR + delta
            power = 0
            for _ in range(n_sim):
                acc_c = rr.binomial(n_per.astype(np.int64), true_a) / n_per
                obs = float((acc_c * n_per).sum() / n_per.sum())
                bs = rr.choice(acc_c, size=(200, len(acc_c)), replace=True).mean(1)
                se = float(np.std(bs))
                if se > 0 and (obs - FLOOR) / se > 1.645:
                    power += 1
            if power / n_sim >= 0.80:
                return float(delta)
        return 0.66

    mde_v = mde(corr, ci_arr)
    print('[MDE] 内容簇口径，alpha=.05 power=.80 最小可测提升 = %+.3f（相对地板）' % mde_v)

    # ---------- J4 字符词袋基线 / J5 长度基线 / 词法审计 ----------
    # 决定性问题：探针读到的是「关系」，还是「我写的谓语词不同」？
    # 若字符词袋在同口径下也能到 J1 的水平，则隐藏态探针没有超出口面词法。
    vcnt = {}
    for r_ in rows:
        for ch in set(r_['text']):
            vcnt[ch] = vcnt.get(ch, 0) + 1
    keep = sorted([ch for ch, v in vcnt.items() if v >= 3])
    cidx = {ch: i for i, ch in enumerate(keep)}
    Xc = np.zeros((n_sent, len(keep)), dtype=np.float32)
    for i, r_ in enumerate(rows):
        for ch in set(r_['text']):
            if ch in cidx:
                Xc[i, cidx[ch]] += 1.0
    Xlen = np.array([[len(r_['text']), len(r_['text']) - len(CONTENTS[r_['ci']]['a'])]
                     for r_ in rows], dtype=np.float32)

    def cv_plain(Xm):
        pred_ = np.full(n_sent, -1, dtype=int)
        for tr_c, te_c in folds:
            rtr, rte = rows_of(tr_c), rows_of(te_c)
            Ztr, Zte = _std_fit(Xm[rtr], Xm[rte])
            Y = np.zeros((len(rtr), 3))
            Y[np.arange(len(rtr)), y[rtr]] = 1.0
            A = _dual(Ztr, Y, [alpha_rel * len(rtr)])[alpha_rel * len(rtr)]
            pred_[rte] = (Zte @ (Ztr.T @ A)).argmax(1)
        return pred_

    pred_j4 = cv_plain(Xc)
    pred_j5 = cv_plain(Xlen)
    j4 = float((pred_j4 == y).mean())
    j5 = float((pred_j5 == y).mean())
    print('[J4] 字符词袋基线（留出内容，%d 字）acc=%.4f' % (len(keep), j4))
    print('[J5] 长度基线（2 维）acc=%.4f' % j5)

    NEG = ['不', '没', '未', '别', '非']
    neg_rate = {}
    for ri, rname in enumerate(RELS):
        mm = np.where(y == ri)[0]
        fr = [1.0 if any(g in rows[i]['text'][len(CONTENTS[rows[i]['ci']]['a']):] for g in NEG) else 0.0
              for i in mm]
        neg_rate[rname] = round(float(np.mean(fr)), 3)
    print('[词法审计] B 从句含否定字比例 %s' % neg_rate)

    # J6：单一「B 含否定字」二元特征基线（关系是否只是否定？）
    X6 = np.array([[1.0 if any(g in rows[i]['text'][len(CONTENTS[rows[i]['ci']]['a']):]
                               for g in NEG) else 0.0] for i in range(n_sent)], dtype=np.float32)
    pred_j6 = cv_plain(X6)
    j6 = float((pred_j6 == y).mean())
    print('[J6] 单一否定字特征 acc=%.4f' % j6)

    # 配对检验：隐藏态探针 vs 字符词袋（McNemar 精确二项）
    import math
    a_ok = pred_lstar == y
    b_ok = pred_j4 == y
    nb = int(np.sum(a_ok & ~b_ok))
    nc = int(np.sum(~a_ok & b_ok))
    nn = nb + nc
    if nn == 0:
        pmc = 1.0
    else:
        k = min(nb, nc)
        tail = sum(math.comb(nn, j) for j in range(0, k + 1)) / (2.0 ** nn)
        pmc = min(1.0, 2.0 * tail)
    print('[对比] 隐藏态 %d 对 / 词袋 %d 对；不一致 %d vs %d；McNemar p=%.4g'
          % (int(a_ok.sum()), int(b_ok.sum()), nb, nc, pmc))

    # 描述性几何：全量拟合（非留出，仅描述），3 类质心在判别平面里的布局
    l_geo = 16
    Xg = feats['P_END'][l_geo]
    Zg, _ = _std_fit(Xg, Xg)
    Yg = np.zeros((n_sent, 3))
    Yg[np.arange(n_sent), y] = 1.0
    Ag = _dual(Zg, Yg, [alpha_rel * n_sent])[alpha_rel * n_sent]
    Wg = Zg.T @ Ag
    Cg = np.stack([Wg.T @ Zg[y == k].mean(0) for k in range(3)], 0)   # 3 x 3
    Cc = Cg - Cg.mean(0, keepdims=True)
    U, sv, Vt = np.linalg.svd(Cc, full_matrices=False)
    coords = U[:, :2] * sv[:2]
    var1 = float(sv[0] ** 2 / (sv ** 2).sum())
    ang = {}
    for i in range(3):
        for j in range(i + 1, 3):
            v1, v2 = coords[i], coords[j]
            cs = float(v1 @ v2 / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-12))
            ang['%s-%s' % (RELS[i], RELS[j])] = round(cs, 3)
    print('[几何·描述性] 层%d 质心二维布局 轴1占比 %.3f  两两 cos(相对原点) %s'
          % (l_geo, var1, ang))
    print('              坐标 %s' % {RELS[k]: [round(float(x), 3) for x in coords[k]] for k in range(3)})

    mde_cons = mde(corr, ci_arr, n_sim=300, force_n=1)
    print('[MDE] 保守口径（内容内完全相关，每簇 1 个独立样本）= %+.3f' % mde_cons)
    acc_c_cons = np.array([corr[ci_arr == g].mean() for g in np.unique(ci_arr)])
    print('[D5] 内容级准确率分布 min/中位/max = %.3f / %.3f / %.3f'
          % (acc_c_cons.min(), np.median(acc_c_cons), acc_c_cons.max()))

    # ---------- 判决 ----------
    def band(acc, ci_, th_pass):
        if ci_[0] <= FLOOR:
            return 'NULL'
        return 'PASS' if acc >= th_pass else 'WEAK'

    v_j1 = band(acc_J1, ci_j1, TH_J1_PASS)
    v_j2 = band(j2_main, ci2, TH_J2_PASS)
    if j3[1] >= j3[3] - TH_J3_TOL:
        v_j3 = 'SHARED_AXIS'
    elif j3[2] >= j3[3] - TH_J3_TOL:
        v_j3 = 'PARTIAL_SHARING'
    else:
        v_j3 = 'PRIVATE_DIRECTIONS'

    if j0_max_dev > 1e-6:
        verdict = 'PIPELINE_LEAK'
    elif acc_J1 < TH_J1_FRAME:
        verdict = 'FRAMEWORK_INVALID'
    elif v_j1 == 'NULL':
        verdict = 'NO_DECODABLE_RELATION'
    elif v_j1 == 'WEAK' and mde_v > (acc_J1 - FLOOR):
        verdict = 'UNDERPOWERED'
    elif v_j1 == 'PASS' and v_j2 == 'PASS':
        verdict = 'SHARED_RELATION_AXIS' if v_j3 == 'SHARED_AXIS' else 'DECODABLE_NOT_SHARED'
    elif v_j1 == 'PASS':
        verdict = 'SURFACE_BOUND'
    else:
        verdict = 'WEAK_ONLY'
    stimuli_unverified = bool(g0b_deb < TH_G0_GRAY)

    per_rel = {}
    per_surf = {}
    for ri, r in enumerate(RELS):
        m = np.where(y == ri)[0]
        per_rel[r] = round(float((pred_lstar[m] == y[m]).mean()), 4)
    for sf in SURFS:
        m = np.where(sf_arr == sf)[0]
        per_surf[sf] = round(float((pred_lstar[m] == y[m]).mean()), 4)

    print()
    print('=' * 68)
    print('判决 J1=%s  J2=%s  J3=%s' % (v_j1, v_j2, v_j3))
    print('总判决 = %s' % verdict)
    print('=' * 68)

    payload = dict(
        stage='full', n_sent=n_sent, n_contents=n_cont, n_layers=L, hidden=H,
        alpha_rel=alpha_rel, lstar=lstar.tolist(),
        g0_acc=round(g0_acc, 4), g0_hist=g0_hist,
        g0b_raw=round(g0b_raw, 4), g0b_deb=round(g0b_deb, 4),
        g0b_per_rel=g0b_per_rel, stimuli_unverified=stimuli_unverified,
        j0={k: round(v, 6) for k, v in j0.items()}, j0_max_dev=round(j0_max_dev, 8),
        J1_acc=round(acc_J1, 4), J1_ci=[round(ci_j1[0], 4), round(ci_j1[1], 4)],
        J1_gap_floor=round(acc_J1 - FLOOR, 4), J1_verdict=v_j1,
        J2_main=round(j2_main, 4), J2_ci=[round(ci2[0], 4), round(ci2[1], 4)],
        J2_reverse=round(j2_rev, 4), J2_punct=round(j2_punct, 4),
        J2_verdict=v_j2, P_CONN_acc=round(acc_conn, 4),
        J3={int(k): round(v, 4) for k, v in j3.items()}, J3_verdict=v_j3,
        null_mean=round(null_mean, 4), null_p95=round(null_p95, 4), null_runs=null_accs,
        mde=round(mde_v, 3), mde_conservative=round(mde_cons, 3),
        J4_char_bag=round(j4, 4), J4_vocab=len(keep), J5_length=round(j5, 4),
        J6_negation=round(j6, 4), mcnemar={'j1_only': nb, 'j4_only': nc, 'p': round(pmc, 6)},
        geom={'layer': l_geo, 'axis1_share': round(var1, 4), 'cos': ang,
              'coords': {RELS[k]: [round(float(x), 3) for x in coords[k]] for k in range(3)}},
        neg_rate=neg_rate,
        acc_by_content=[round(float(x), 3) for x in acc_c_cons],
        per_rel=per_rel, per_surf=per_surf,
        profile_P_END=[round(float(x), 4) for x in prof['P_END']],
        profile_P_CONN=[round(float(x), 4) for x in prof['P_CONN']],
        profile_P_AE=[round(float(x), 4) for x in prof['P_AE']],
        fold_layer_scores=[[round(float(x), 4) for x in fold_scores[i].tolist()]
                           for i in range(N_OUTER)],
        verdict=verdict,
        pred_export=dict(
            labels=y.tolist(), content=ci_arr.tolist(), surf=sf_arr.tolist(),
            P_END=pred_lstar.tolist(), P_AE=pred_store['P_AE'].tolist(),
            P_CONN=pred_store['P_CONN'].tolist(),
            J2=j2_pred.tolist(), J2_hit=j2_hit.astype(int).tolist(),
            J3={int(k): j3_pred[k].tolist() for k in (1, 2, 3)},
            J4=pred_j4.tolist(), J5=pred_j5.tolist(), J6=pred_j6.tolist(),
        ),
        texts=[r['text'] for r in rows],
        elapsed_s=round(time.time() - t0, 1),
    )
    out = provenance.write_result('pat_q7a', payload, model_path=MODEL_PATH,
                                  seed=SEED, tool='tools/pat_q7a.py')
    print('落盘 %s' % out)
    print('用时 %.0fs' % (time.time() - t0))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
