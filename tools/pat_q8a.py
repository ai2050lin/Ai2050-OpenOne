# -*- coding: utf-8 -*-
"""Q8-A：静态嵌入的系统解剖。

预注册：research/gpt5/docs/PAT_Q8A_PREREG_v1.md（§1–§7 冻结，§8 跑后修订）

P1 单位分层     词表按 整词/单字/字节回退 分成三个几何种群（字节塌缩 +0.15）
P2 频率—范数律  log‖e‖ ~ zipf，符号为正，留出折 Spearman ≥ 0.30，8 段单调
P3 类别骨架     within(FRUIT) − within(FREQ_CTRL) ≥ +0.05；留一 2AFC ≥ 0.80；共享字过滤
P4 色相几何     Mantel ρ(色相圆周距离, cos 距离) ≥ 0.30
P5 生僻字=字节  覆盖 ≥ 30%；码位距离 ↔ 相似度 ρ ≥ +0.50；within(RARE)−within(COMMON) ≥ +0.10
P6 形旁         8 组 within−between ≥ +0.03；异质组(石) 与 同质组 对比
P7 单坐标载荷   坐标 η² 普查；top-k 留出曲线（k=32 时 ≥ 0.75）；CO/FR 主干重叠
P8 嵌入→输出    偏 Spearman(cos, logprob | zipf) ≥ 0.30

用法
----
  SMOKE=1 .venv/Scripts/python.exe -X utf8 tools/pat_q8a.py
  .venv/Scripts/python.exe -X utf8 tools/pat_q8a.py
  MODEL=D:/develop/model/ZhipuAI/glm-4-9b-chat-hf .venv/Scripts/python.exe -X utf8 tools/pat_q8a.py
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
for _p in (str(ROOT), str(ROOT / 'tools')):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import provenance

SMOKE = os.environ.get('SMOKE', '') == '1'
MODEL_PATH = os.environ.get('MODEL') or 'D:/develop/model/Qwen/Qwen3-4B'
SEED = 20261101
RNG = np.random.default_rng(SEED)

N_PERM = 2000 if not SMOKE else 200
N_PERM_MANTEL = 10000 if not SMOKE else 500
N_PERM_COORD = 200 if not SMOKE else 40

LOG = []


def say(*a):
    line = ' '.join(str(x) for x in a)
    print(line, flush=True)
    LOG.append(line)


# ============================ 材料（预注册 §1） ============================

COLOR_HEX = {
    '红': 'FF0000', '大红': 'E60000', '朱红': 'FF3300', '绯红': 'D0202F', '桃红': 'FFB3C6',
    '粉红': 'FFB6C1', '洋红': 'FF00AA', '玫瑰红': 'C21E56', '酒红': '7B1E28', '枣红': '8B2E2E',
    '橘红': 'FF4500', '橙': 'FF8000', '橘黄': 'FFA500', '金黄': 'FFD700', '黄': 'FFFF00',
    '柠檬黄': 'FFF44F', '米黄': 'F5DEB3', '土黄': 'C8A165', '草绿': '6B8E23', '嫩绿': 'B4F0A0',
    '翠绿': '00C957', '绿': '00A000', '墨绿': '014421', '橄榄绿': '808000', '碧绿': '4E9F3D',
    '湖蓝': '30A5C8', '天蓝': '87CEEB', '浅蓝': 'ADD8E6', '宝蓝': '1F51A0', '蓝': '0000FF',
    '藏青': '2E4A7D', '靛蓝': '4B0082', '青': '00A0A0', '青绿': '00B389', '紫': '8000FF',
    '紫罗兰': '8A2BE2', '丁香紫': 'B39EB5', '葡萄紫': '5B2C6F', '粉': 'FFC0CB', '棕': '8B4513',
    '咖啡色': '6F4E37', '巧克力色': '4E342E', '卡其色': 'C3B091', '驼色': 'C19A6B', '灰': '808080',
    '银灰': 'C0C0C0', '铁灰': '52595D', '烟灰': '7E7E7E', '黑': '000000', '雪白': 'FFFAFA',
    '白': 'FFFFFF', '米色': 'F5F5DC',
}

FRUIT = ['苹果', '香蕉', '橙子', '橘子', '柑', '柠檬', '柚子', '葡萄', '草莓', '蓝莓', '树莓',
         '西瓜', '甜瓜', '哈密瓜', '桃子', '梨', '李子', '杏', '樱桃', '芒果', '菠萝', '木瓜',
         '荔枝', '龙眼', '榴莲', '山竹', '火龙果', '猕猴桃', '柿子', '石榴', '无花果', '杨梅',
         '桑葚', '枇杷', '山楂', '椰子', '牛油果', '百香果', '杨桃', '莲雾', '枣', '橄榄',
         '板栗', '核桃', '榛子', '腰果', '花生', '番石榴', '蜜瓜', '蔓越莓']
FRUIT_NUT = ['板栗', '核桃', '榛子', '腰果', '花生']

RARE_CURATED = list('龘靐齉齾爨爩麤麣鱻龖龗灪羴猋骉毳掱嫑兲巭勥埊玊烎氼嘦圕孖冇乜冚蠱彧翀仝亓甪氍觳黼黻鼗龠龢籴粜耷耨薅搴殳厍卺耄耋饕餮魑魅鬻龃龉龊龌黩黥黢黟黠鼬鼹鼷鼯齑')

COMMON_CHAR = list('的一是不了在人有我他这个们中来上大为和国地到以说时要就出会可也你对生能而子那得于着下自之年过发后作里用道行所然家种事成方多经去法学')

RADICAL = {
    '氵': '江河湖海洋波浪流清洗泳溪池滩湾潮',
    '木': '松柏杨柳枫桃梅桂桐榆槐枝根树林桥楼',
    '火': '烧烤炒炸煮蒸烙炼燃焰灯炉炮燥',
    '讠': '说话讲谈论议评谢请谁词语读记识训访询',
    '忄': '情恨惧怕悔忙慢懂慌怖惊悦愁恰惭愧',
    '石': '硬碰碗破碎砖砂码磁确',
    '虫': '蚊蝇蝶蝠蜂蚁蛇虾蟹蜘蛛蜜蜡',
    '鸟': '鸡鸭鹅鹤鸦鸽鹰燕鹏鹊',
}
RAD_HOMOG = {'氵', '木', '火', '鸟', '虫'}
RAD_HETERO = {'石', '讠', '忄'}


# ============================ 数值工具 ============================

def l2n(X):
    n = np.linalg.norm(X, axis=-1, keepdims=True)
    return X / np.maximum(n, 1e-9)


def cosine(A, B):
    return l2n(A) @ l2n(B).T


def offdiag_mean(C):
    m = ~np.eye(C.shape[0], dtype=bool)
    return float(C[m].mean()) if m.any() else float('nan')


def within_mean(X):
    """组内平均两两余弦（中心化向量由调用方传入）。"""
    if len(X) < 2:
        return float('nan')
    return offdiag_mean(cosine(X, X))


def between_mean(X, Y):
    return float(cosine(X, Y).mean())


def spearman(a, b):
    return float(stats.spearmanr(a, b).statistic)


def wilson(k, n, z=1.96):
    if n == 0:
        return (float('nan'), float('nan'))
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def mde_spearman(n, alpha=0.05):
    """给定 n，alpha=0.05 双尾可检出的最小 |ρ|（Fisher z 近似）。"""
    if n < 4:
        return float('nan')
    z_crit = 1.959964 + 0.6744898  # 80% power
    return float(np.tanh(z_crit / np.sqrt(n - 3)))


def partial_spearman(x, y, z):
    """控制 z 的偏 Spearman（秩残差法）。"""
    rx, ry, rz = (stats.rankdata(v) for v in (x, y, z))
    Z = np.column_stack([np.ones_like(rz), rz])
    bx = np.linalg.lstsq(Z, rx, rcond=None)[0]
    by = np.linalg.lstsq(Z, ry, rcond=None)[0]
    ex, ey = rx - Z @ bx, ry - Z @ by
    return float(np.corrcoef(ex, ey)[0, 1])


def srgb_to_lab(hexstr):
    """标准 sRGB(D65) → CIELAB；返回 (L*, C*, h°) 与 (a*, b*)。"""
    r, g, b = (int(hexstr[i:i + 2], 16) / 255.0 for i in (0, 2, 4))
    lin = []
    for u in (r, g, b):
        lin.append(u / 12.92 if u <= 0.04045 else ((u + 0.055) / 1.055) ** 2.4)
    rl, gl, bl = lin
    X = rl * 0.4124564 + gl * 0.3575761 + bl * 0.1804375
    Y = rl * 0.2126729 + gl * 0.7151522 + bl * 0.0721750
    Z = rl * 0.0193339 + gl * 0.1191920 + bl * 0.9503041
    Xn, Yn, Zn = 0.95047, 1.0, 1.08883

    def f(t):
        return t ** (1 / 3) if t > 0.008856 else (7.787 * t + 16 / 116)

    fx, fy, fz = f(X / Xn), f(Y / Yn), f(Z / Zn)
    L = 116 * fy - 16
    a = 500 * (fx - fy)
    bb = 200 * (fy - fz)
    C = float(np.hypot(a, bb))
    h = float(np.degrees(np.arctan2(bb, a)) % 360)
    return float(L), C, h, float(a), float(bb)


def hue_dist(h1, h2):
    d = np.abs(np.asarray(h1)[:, None] - np.asarray(h2)[None, :])
    return np.minimum(d, 360 - d)


def mantel(D1, D2, n_perm=N_PERM_MANTEL):
    """Mantel：上三角 Spearman。返回 (rho, p_perm)。"""
    iu = np.triu_indices(D1.shape[0], 1)
    a, b = D1[iu], D2[iu]
    rho = spearman(a, b)
    cnt = 0
    perm = RNG.permutation(D1.shape[0]) if n_perm else None
    for _ in range(n_perm):
        p = RNG.permutation(D1.shape[0])
        r = spearman(D1[np.ix_(p, p)][iu], b)
        if abs(r) >= abs(rho):
            cnt += 1
    return float(rho), (cnt + 1) / (n_perm + 1)


# ============================ 模型与词向量 ============================

def load_model():
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL_PATH)
    kw = dict(dtype=torch.bfloat16)
    if os.environ.get('DEVICE_MAP'):
        kw['device_map'] = os.environ['DEVICE_MAP']
    model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, **kw)
    model.eval()
    E = model.get_input_embeddings().weight.detach().float().cpu().numpy()
    return tok, model, E


def build_sets(tok, E):
    """按 §1.1 生成词向量表：A 档（单 token）与 B 档（token 均值）。"""
    dec = [tok.decode([i]) for i in range(E.shape[0])]

    def vec(ids, mode):
        if mode == 'A' and len(ids) != 1:
            return None
        return E[ids].mean(0)

    def cjk(s):
        return len(s) > 0 and all('\u3400' <= ch <= '\u9fff' or '\u4e00' <= ch <= '\u9fff'
                                  or '\uf900' <= ch <= '\ufaff' for ch in s)

    sets = {}

    def add(name, words):
        ids_a, ids_b, used = [], [], []
        for w in words:
            ids = tok.encode(w, add_special_tokens=False)
            if not ids:
                continue
            used.append(w)
            ids_a.append(ids[0] if len(ids) == 1 else None)
            ids_b.append(ids)
        sets[name] = dict(words=used, ids_a=ids_a, ids_b=ids_b,
                          n_single=sum(1 for x in ids_a if x is not None))
        return sets[name]

    add('COLOR', list(COLOR_HEX.keys()))
    add('FRUIT', FRUIT)
    add('RARE', RARE_CURATED)
    add('COMMON', COMMON_CHAR)
    for k, v in RADICAL.items():
        add('RAD' + k, list(v))
    return sets, dec, cjk


# ============================ P0 近邻示例（描述性） ============================

def p0_neighbors(E, mu, dec, sets, probes):
    """在全部词表行里找 top-8 近邻（中心化 cos）。纯描述，不作判决。"""
    from wordfreq import zipf_frequency as zf
    G = l2n(E - mu)
    out = {}
    for w in probes:
        ids = TOK.encode(w, add_special_tokens=False)
        v = E[ids].mean(0) - mu
        v = v / max(np.linalg.norm(v), 1e-9)
        sims = G @ v
        order = np.argsort(-sims)[:9]
        nb = []
        for j in order:
            t = dec[j]
            if t == w:
                continue
            nb.append(dict(tok=t, sim=round(float(sims[j]), 4),
                           zipf=round(float(zf(t, 'zh')), 2)))
            if len(nb) == 8:
                break
        out[w] = dict(n_tok=len(ids), neighbors=nb)
        say('P0 「%s」(%d token) 近邻: ' % (w, len(ids)) +
            ' | '.join('%s %.3f (zipf %.1f)' % (d['tok'], d['sim'], d['zipf']) for d in nb))
    return out


# ============================ P9 中英对译秩（跑后追加，探索） ============================

PAIRS = [('红', 'red'), ('蓝', 'blue'), ('绿', 'green'), ('黄', 'yellow'), ('黑', 'black'),
         ('白', 'white'), ('爱', 'love'), ('水', 'water'), ('火', 'fire'), ('山', 'mountain'),
         ('猫', 'cat'), ('狗', 'dog'), ('鸟', 'bird'), ('鱼', 'fish'), ('门', 'door'),
         ('手', 'hand'), ('眼', 'eye'), ('心', 'heart'), ('车', 'car'), ('书', 'book'),
         ('钱', 'money'), ('天', 'sky'), ('雨', 'rain'), ('雪', 'snow'), ('风', 'wind'),
         ('树', 'tree'), ('花', 'flower'), ('草', 'grass'), ('血', 'blood'), ('睡', 'sleep'),
         ('吃', 'eat'), ('跑', 'run'), ('三', 'three'), ('大', 'big'), ('小', 'small'),
         ('好', 'good'), ('新', 'new'), ('冷', 'cold'), ('热', 'hot'), ('快', 'fast')]


def p9_translation(E, mu, dec, tok):
    """每个汉字的最佳英文对译 token 在 151936 行里的名次。零基线 = 随机英文词的名次。"""
    G = l2n(E - mu)
    ascii_ids = np.array([i for i, t in enumerate(dec)
                          if t.strip() and t.strip().isascii() and t.strip().isalpha()])
    rows = []
    for zh, en in PAIRS:
        ids = tok.encode(zh, add_special_tokens=False)
        if len(ids) != 1:
            continue
        v = E[ids[0]] - mu
        v = v / max(np.linalg.norm(v), 1e-9)
        sims = G @ v
        sims[ids[0]] = -9.0          # 排除自身（否则自身恒占第 1 名）
        order = np.argsort(-sims)
        rank_of = {int(j): r + 1 for r, j in enumerate(order)}
        variants = {en, ' ' + en, en.capitalize(), ' ' + en.capitalize(),
                    en.upper(), '_' + en, '-' + en}
        best, best_tok = 10 ** 9, ''
        for w in variants:
            vip = tok.encode(w, add_special_tokens=False)
            if len(vip) != 1:
                continue
            r = rank_of[int(vip[0])]
            if r < best:
                best, best_tok = r, w
        # 零基线：5 个随机英文词的同样口径名次（同一汉字向量）
        nulls = []
        for j in RNG.choice(ascii_ids, 5, replace=False):
            s0 = sims[j]
            nulls.append(int((sims > s0).sum()) + 1)
        rows.append(dict(zh=zh, en=en, rank=best, tok=best_tok,
                         null_median=float(np.median(nulls)), sim=float(sims.max())))
    ranks = np.array([r['rank'] for r in rows])
    nulls = np.array([r['null_median'] for r in rows])
    out = dict(n=len(rows), median_rank=float(np.median(ranks)),
               frac_top1=float((ranks == 1).mean()), frac_top5=float((ranks <= 5).mean()),
               frac_top10=float((ranks <= 10).mean()), frac_top100=float((ranks <= 100).mean()),
               null_median=float(np.median(nulls)),
               rows=sorted(rows, key=lambda r: r['rank']))
    say('P9 中英对译秩 n=%d：中位名次 %.1f（随机英文词基线 %.0f）；top1 %.2f / top5 %.2f / top10 %.2f / top100 %.2f'
        % (out['n'], out['median_rank'], out['null_median'], out['frac_top1'],
           out['frac_top5'], out['frac_top10'], out['frac_top100']))
    say('P9 最好 8 个: ' + ' | '.join('%s→%s #%d' % (r['zh'], r['tok'].strip(), r['rank'])
                                     for r in out['rows'][:8]))
    bad = [r for r in out['rows'] if r['rank'] > 1000]
    say('P9 名次 >1000 的 %d 个: %s' % (len(bad), ' '.join('%s(%d)' % (r['zh'], r['rank'])
                                                          for r in bad)))
    return out


# ============================ P10 简繁复本（跑后追加，探索） ============================

SIMP_TRAD = [('红', '紅'), ('爱', '愛'), ('说', '說'), ('门', '門'), ('车', '車'), ('书', '書'),
             ('国', '國'), ('学', '學'), ('见', '見'), ('语', '語'), ('时', '時'), ('过', '過'),
             ('这', '這'), ('来', '來'), ('会', '會'), ('开', '開'), ('后', '後'), ('关', '關'),
             ('长', '長'), ('问', '問'), ('间', '間'), ('点', '點'), ('电', '電'), ('头', '頭'),
             ('风', '風'), ('飞', '飛'), ('马', '馬'), ('鸟', '鳥'), ('鱼', '魚'), ('龙', '龍'),
             ('卖', '賣'), ('买', '買'), ('读', '讀'), ('听', '聽'), ('无', '無'), ('为', '為')]


def p10_simp_trad(E, mu, dec, tok):
    """简繁对：cos 与最佳名次。若名次恒为 1，说明词表为同一字存了两份几乎相同的行。"""
    G = l2n(E - mu)
    rows = []
    for s_, t_ in SIMP_TRAD:
        a = tok.encode(s_, add_special_tokens=False)
        b = tok.encode(t_, add_special_tokens=False)
        if len(a) != 1 or len(b) != 1:
            continue
        va = E[a[0]] - mu
        va = va / max(np.linalg.norm(va), 1e-9)
        sims = G @ va
        sims[a[0]] = -9.0            # 排除自身
        r = int((sims > sims[b[0]]).sum()) + 1
        rows.append(dict(simp=s_, trad=t_, cos=float(sims[b[0]]), rank=r))
    cos = np.array([r['cos'] for r in rows])
    rk = np.array([r['rank'] for r in rows])
    out = dict(n=len(rows), cos_median=float(np.median(cos)), cos_min=float(cos.min()),
               cos_max=float(cos.max()), frac_rank1=float((rk == 1).mean()),
               rows=rows)
    say('P10 简繁对 n=%d：cos 中位 %.4f [%.3f, %.3f]；繁体形式排第 1 的比例 %.2f'
        % (len(rows), out['cos_median'], out['cos_min'], out['cos_max'], out['frac_rank1']))
    worst = sorted(rows, key=lambda r: r['cos'])[:5]
    say('P10 最不像的 5 对: ' + ' | '.join('%s/%s %.3f #%d' % (r['simp'], r['trad'], r['cos'], r['rank'])
                                          for r in worst))
    return out


# ============================ P1 单位分层 ============================

def p1_unit_types(E, dec, cjk):
    """词表按单元类型分层。fragment = 解出来是 '\ufffd' 的字节残片（非合法 UTF-8）。"""
    n = E.shape[0]
    types = np.empty(n, dtype=object)
    for i, s in enumerate(dec):
        if '\ufffd' in s:
            types[i] = 'fragment'
        elif len(s) == 1 and cjk(s):
            types[i] = 'char'
        elif len(s) >= 2 and cjk(s):
            types[i] = 'word'
        else:
            types[i] = 'other'
    norms = np.linalg.norm(E, axis=1)
    mu = E.mean(0)
    out = {}
    for t in ('word', 'char', 'fragment', 'other'):
        idx = np.where(types == t)[0]
        if len(idx) == 0:
            say('P1 %-8s 不存在' % t)
            continue
        sub = idx if len(idx) <= 1500 else RNG.choice(idx, 1500, replace=False)
        Xc = E[sub] - mu
        out[t] = dict(n=int(len(idx)), n_sampled=int(len(sub)),
                      norm_median=float(np.median(norms[idx])),
                      norm_mean=float(norms[idx].mean()),
                      within_cos_centered=within_mean(Xc),
                      within_cos_raw=within_mean(E[sub]))
        say('P1 %-8s n=%6d  med‖e‖=%.4f  within_cos(centered)=%.4f  (raw %.4f)'
            % (t, len(idx), out[t]['norm_median'], out[t]['within_cos_centered'],
               out[t]['within_cos_raw']))
    gap = out['fragment']['within_cos_centered'] - out['word']['within_cos_centered']
    out['fragment_minus_word'] = float(gap)
    out['verdict'] = 'BYTE_COLLAPSE' if gap >= 0.15 else 'NO_BYTE_COLLAPSE'
    # 抽样零分布：同规模 word 子集重抽，看 fragment 的 within 是否超出 word 的抽样波动
    word_idx = np.where(types == 'word')[0]
    nb = out['fragment']['n_sampled']
    nulls = []
    for _ in range(200 if not SMOKE else 30):
        s = RNG.choice(word_idx, min(nb, len(word_idx)), replace=False)
        nulls.append(within_mean(E[s] - mu))
    nulls = np.array(nulls)
    out['word_subset_null'] = dict(mean=float(nulls.mean()), p95=float(np.percentile(nulls, 95)))
    out['fragment_p_vs_word_null'] = float((nulls >= out['fragment']['within_cos_centered']).mean())
    say('P1 残片−整词 = %+.4f  (word 子集零分布 %d 次：均 %.4f, p95 %.4f)  => %s'
        % (gap, len(nulls), nulls.mean(), np.percentile(nulls, 95), out['verdict']))
    return out, types


# ============================ P2 频率—范数律 ============================

def p2_freq_norm(E, dec, cjk, types):
    from wordfreq import zipf_frequency as zf
    idx = np.where(types == 'char')[0]
    chars, fs, ns = [], [], []
    for i in idx:
        s = dec[i]
        f = zf(s, 'zh')
        if f > 0:
            chars.append(s)
            fs.append(f)
            ns.append(float(np.linalg.norm(E[i])))
    chars, fs, ns = np.array(chars), np.array(fs), np.array(ns)
    y = np.log(ns)
    rho = spearman(y, fs)
    perm = np.array([spearman(y, RNG.permutation(fs)) for _ in range(N_PERM)])
    p_perm = float((np.abs(perm) >= abs(rho)).mean())

    # 5 折按字符划分的留出预测（等距线性回归）
    order = RNG.permutation(len(chars))
    folds = np.array_split(order, 5)
    preds = np.zeros(len(chars))
    for f in folds:
        tr = np.setdiff1d(order, f)
        A = np.column_stack([np.ones(len(tr)), fs[tr]])
        b = np.linalg.lstsq(A, y[tr], rcond=None)[0]
        preds[f] = np.column_stack([np.ones(len(f)), fs[f]]) @ b
    rho_hold = spearman(preds, y)

    # 8 段单调性
    q = np.quantile(fs, np.linspace(0, 1, 9))
    bands = []
    for i in range(8):
        m = (fs >= q[i]) & (fs <= q[i + 1])
        bands.append(dict(band=i, zipf_lo=float(q[i]), zipf_hi=float(q[i + 1]),
                          n=int(m.sum()), norm_median=float(np.median(ns[m])),
                          lognorm_median=float(np.median(y[m]))))
    band_rho = spearman([b['lognorm_median'] for b in bands], [b['zipf_lo'] for b in bands])
    out = dict(n=int(len(chars)), rho=rho, p_perm=p_perm, rho_holdout=rho_hold,
               bands=bands, band_rho=band_rho, mde=mde_spearman(len(chars)))
    cond = (rho > 0 and p_perm < 1e-4, rho_hold >= 0.30, band_rho > 0)
    out['cond'] = dict(pos_and_sig=bool(cond[0]), holdout=bool(cond[1]), monotone=bool(cond[2]))
    # D16：形状判据必须能表达"有趋势但未达阈"与"非单调峰值"
    meds = np.array([b['lognorm_median'] for b in bands])
    peak = int(np.argmax(meds))
    last_vs_peak = float(meds[-1] - meds[peak])
    low_reverse = bool(meds[0] < meds[1] and band_rho > 0)  # 低端同向，无 U
    out['shape'] = dict(peak_band=peak, peak_zipf=float(bands[peak]['zipf_lo']),
                        last_minus_peak=last_vs_peak,
                        interior_peak=bool(0 < peak < len(meds) - 1))
    if all(cond):
        out['verdict'] = 'PASS'
    elif cond[0] and not cond[1]:
        out['verdict'] = ('PEAK_AT_MID_FREQUENCY' if out['shape']['interior_peak']
                          else ('WEAK_MONOTONE' if band_rho > 0 else 'NON_MONOTONE'))
    else:
        out['verdict'] = 'FAIL'
    say('P2 n=%d  ρ_s(log‖e‖,zipf)=%+.4f (perm p=%.4g, MDE %.3f)  留出 ρ=%+.4f  8 段 ρ=%+.4f  => %s'
        % (len(chars), rho, p_perm, out['mde'], rho_hold, band_rho, out['verdict']))
    say('P2 八段 (zipf_lo, lognorm_med, n): ' +
        ' '.join('(%.2f,%.3f,%d)' % (b['zipf_lo'], b['lognorm_median'], b['n']) for b in bands))
    return out


# ============================ P3 类别骨架 ============================

def freq_matched_pool(dec, cjk, types, lo=1.0, hi=6.0):
    from wordfreq import zipf_frequency as zf
    idx = np.where(types == 'word')[0]
    pool = []
    for i in idx:
        s = dec[i]
        if len(s) == 2 and cjk(s):
            f = zf(s, 'zh')
            if lo <= f <= hi:
                pool.append((s, int(i), float(f)))
    return pool


def shared_char(a, b):
    return len(set(a) & set(b)) > 0


def p3_category(E, mu, sets, pool, sm=True):
    from wordfreq import zipf_frequency as zf
    tok = TOK

    def vecs(name, mode):
        ids = sets[name]['ids_a'] if mode == 'A' else sets[name]['ids_b']
        ws = [w for w, x in zip(sets[name]['words'], ids) if x is not None]
        if mode == 'A':
            xs = [E[x] for w, x in zip(sets[name]['words'], ids) if x is not None]
        else:
            xs = [E[x].mean(0) for w, x in zip(sets[name]['words'], ids) if x is not None]
        return ws, np.array(xs) - mu

    Wf, Fb = vecs('FRUIT', 'B')
    Wc, Cb = vecs('COLOR', 'B')
    WfA, FbA = vecs('FRUIT', 'A')
    WcA, CbA = vecs('COLOR', 'A')

    w_fruit = within_mean(Fb)
    w_color = within_mean(Cb)
    b_fc = between_mean(Fb, Cb)

    # 频率匹配对照：每个水果配 4 个最近 zipf 的双字词，抽 50 个构成一个对照集
    fs = np.array([zf(w, 'zh') for w in Wf])
    pool_f = np.array([p[2] for p in pool])
    cand = []
    for f in fs:
        cand.append(np.argsort(np.abs(pool_f - f))[:4])
    nulls = []
    for _ in range(N_PERM):
        pick = [pool[int(RNG.choice(c))] for c in cand]
        X = np.array([E[p[1]] for p in pick]) - mu
        nulls.append(within_mean(X))
    nulls = np.array(nulls)
    ctrl_mean = float(nulls.mean())
    p_gap = float((nulls >= w_fruit).mean())
    gap = w_fruit - ctrl_mean

    # 2AFC 留一：B 档
    def afc(A, B, label):
        ok = 0
        for i in range(len(A)):
            ca = A[np.arange(len(A)) != i].mean(0)
            cb = B.mean(0)
            da = 1 - float(cosine(A[i:i + 1], ca[None])[0, 0])
            db = 1 - float(cosine(A[i:i + 1], cb[None])[0, 0])
            ok += int(da < db)
        acc = ok / len(A)
        return dict(label=label, n=len(A), acc=float(acc), ci=list(wilson(ok, len(A))),
                    chance=0.5, mde=mde_spearman(len(A)))

    afc_f = afc(Fb, Cb, 'FRUIT_vs_COLOR')
    afc_c = afc(Cb, Fb, 'COLOR_vs_FRUIT')
    afc_fA = afc(FbA, CbA, 'FRUIT_vs_COLOR_singletoken')
    afc_cA = afc(CbA, FbA, 'COLOR_vs_FRUIT_singletoken')

    # 共享字过滤后重算 within(FRUIT)
    n_pairs = 0
    n_shared = 0
    keep = []
    for i in range(len(Wf)):
        for j in range(i + 1, len(Wf)):
            n_pairs += 1
            if shared_char(Wf[i], Wf[j]):
                n_shared += 1
            else:
                keep.append((i, j))
    Xn = l2n(Fb)
    if keep:
        w_fruit_ns = float(np.mean([float(Xn[i] @ Xn[j]) for i, j in keep]))
    else:
        w_fruit_ns = float('nan')
    drop = (w_fruit - w_fruit_ns) / w_fruit if w_fruit else float('nan')

    # P3d（跑后追加，§8）：把 COLOR / FRUIT 的 zipf 分布按分位对齐后重做留一 2AFC
    zc = np.array([zf(w, 'zh') for w in Wc])
    zff = np.array([zf(w, 'zh') for w in Wf])
    lo, hi = max(zc.min(), zff.min()), min(zc.max(), zff.max())
    sc = [i for i in range(len(zc)) if lo <= zc[i] <= hi]
    sf = [i for i in range(len(zff)) if lo <= zff[i] <= hi]
    kk = min(len(sc), len(sf))
    boot = []
    for _ in range(200 if not SMOKE else 30):
        qs = np.quantile(zc[sc], np.linspace(0, 1, kk + 1))
        ic, ifs = [], []
        for j in range(kk):
            cc = [i for i in sc if qs[j] <= zc[i] <= qs[j + 1]]
            cf = [i for i in sf if qs[j] <= zff[i] <= qs[j + 1]]
            if not cc or not cf:
                continue
            ic.append(int(RNG.choice(cc)))
            ifs.append(int(RNG.choice(cf)))
        A, B = Cb[ic], Fb[ifs]
        if len(A) < 5:
            continue
        ok = 0
        for i in range(len(A)):
            ca = A[np.arange(len(A)) != i].mean(0)
            if (1 - float(cosine(A[i:i + 1], ca[None])[0, 0])) < \
               (1 - float(cosine(A[i:i + 1], B.mean(0)[None])[0, 0])):
                ok += 1
        boot.append(ok / len(A))
    boot = np.array(boot)
    fm = dict(k=int(kk), n_boot=int(len(boot)), acc_mean=float(boot.mean()),
              acc_lo=float(np.percentile(boot, 5)), acc_hi=float(np.percentile(boot, 95)),
              zipf_lo=float(lo), zipf_hi=float(hi))
    fm['verdict'] = 'PASS_FREQ_MATCHED' if fm['acc_mean'] >= 0.80 else 'FAIL'
    say('P3d 频率匹配 2AFC（zipf∈[%.2f,%.2f]，每折 %d vs %d，%d 次）：%.3f [%.3f, %.3f] => %s'
        % (lo, hi, kk, kk, len(boot), fm['acc_mean'], fm['acc_lo'], fm['acc_hi'], fm['verdict']))

    out = dict(n_fruit=len(Wf), n_color=len(Wc), freq_matched=fm,
               within_fruit=w_fruit, within_color=w_color, between=b_fc,
               ctrl_within_mean=ctrl_mean, ctrl_within_p95=float(np.percentile(nulls, 95)),
               gap=float(gap), p_gap=p_gap,
               afc=[afc_f, afc_c, afc_fA, afc_cA],
               share_pairs=dict(n_pairs=n_pairs, n_shared=n_shared,
                                frac=float(n_shared / max(n_pairs, 1)),
                                within_no_shared=w_fruit_ns, drop_frac=float(drop)))
    out['verdict_gap'] = 'PASS' if (gap >= 0.05 and p_gap < 0.01) else 'FAIL'
    out['verdict_afc'] = 'PASS' if afc_f['acc'] >= 0.80 else 'FAIL'
    if drop > 0.5:
        out['verdict_share'] = 'CHAR_OVERLAP_EXPLAINED'
    else:
        out['verdict_share'] = 'SEMANTIC_RESIDUAL'
    say('P3 within(FRUIT)=%.4f within(COLOR)=%.4f between=%.4f  ctrl=%.4f(±)  Δ=%+.4f p=%.4f => %s'
        % (w_fruit, w_color, b_fc, ctrl_mean, gap, p_gap, out['verdict_gap']))
    for a in out['afc']:
        say('P3 2AFC %-26s n=%2d acc=%.4f CI[%.3f,%.3f] MDEρ=%.2f'
            % (a['label'], a['n'], a['acc'], a['ci'][0], a['ci'][1], a['mde']))
    say('P3 共享字 词对 %d/%d (%.0f%%)  去共享字后 within=%.4f  下降 %.1f%% => %s'
        % (n_shared, n_pairs, 100 * n_shared / max(n_pairs, 1), w_fruit_ns, 100 * drop,
           out['verdict_share']))
    return out


# ============================ P4 色相几何 ============================

def p4_hue(E, mu, sets, model, tok):
    import torch
    lab = {w: srgb_to_lab(h) for w, h in COLOR_HEX.items()}
    chrom = [w for w in sets['COLOR']['words'] if lab[w][1] > 10]
    words = sets['COLOR']['words']
    ids = sets['COLOR']['ids_b']
    V = np.array([E[i].mean(0) for i in ids]) - mu
    pos = {w: k for k, w in enumerate(words)}
    sel = [pos[w] for w in chrom]
    Vs = V[sel]
    h = np.array([lab[w][2] for w in chrom])
    L = np.array([lab[w][0] for w in chrom])
    C = np.array([lab[w][1] for w in chrom])

    Dh = hue_dist(h, h)
    Dc = 1 - cosine(Vs, Vs)
    rho_h, p_h = mantel(Dh, Dc)
    rho_L = spearman(L, np.linalg.norm(Vs, axis=1))
    p_L = float((np.abs([spearman(L, RNG.permutation(np.linalg.norm(Vs, axis=1)))
                         for _ in range(N_PERM)]) >= abs(rho_L)).mean())

    # 单 token 稳健性
    single = [w for w in chrom if sets['COLOR']['ids_a'][pos[w]] is not None]
    if len(single) >= 6:
        Vs1 = np.array([E[sets['COLOR']['ids_a'][pos[w]]] for w in single]) - mu
        h1 = np.array([lab[w][2] for w in single])
        rho_h1, p_h1 = mantel(hue_dist(h1, h1), 1 - cosine(Vs1, Vs1),
                              n_perm=2000 if not SMOKE else 300)
    else:
        rho_h1, p_h1 = float('nan'), float('nan')

    # 上下文对照：模板「颜色：{w}。」取 L20 末位
    ctx = {}
    ctx_layer = min(20, model.config.num_hidden_layers)
    try:
        with torch.no_grad():
            reps = []
            for w in chrom:
                txt = '颜色：%s。' % w
                enc = tok(txt, return_tensors='pt')
                o = model(**enc, output_hidden_states=True)
                hs = o.hidden_states[ctx_layer][0, -1].float().cpu().numpy()
                reps.append(hs - mu)
            reps = np.array(reps)
        rho_ctx, p_ctx = mantel(hue_dist(h, h), 1 - cosine(reps, reps),
                                n_perm=2000 if not SMOKE else 300)
        ctx = dict(layer=ctx_layer, rho=rho_ctx, p=p_ctx, n=len(chrom))
    except Exception as e:  # pragma: no cover
        ctx = dict(error=repr(e))

    out = dict(n_chromatic=len(chrom), words=chrom,
               hue=dict(rho=rho_h, p=p_h, mde=mde_spearman(len(chrom))),
               lightness=dict(rho=rho_L, p=p_L, direction='n/a'),
               single_token=dict(n=len(single), rho=rho_h1, p=p_h1),
               contextual=ctx,
               lab={w: dict(L=lab[w][0], C=lab[w][1], h=lab[w][2]) for w in chrom})
    mde_h = mde_spearman(len(chrom))
    if rho_h >= 0.30 and p_h < 0.01:
        out['verdict'] = 'PASS'
    elif mde_h > 0.30:
        # D23：本设计在 80% 功效下只能检出 ≥MDE 的效应，预注册阈 0.30 落在功效之下 ⇒ 不可判决
        out['verdict'] = 'PENDING_power'
    else:
        out['verdict'] = 'FAIL'
    out['verdict_L'] = 'LIGHTNESS_LINKED' if (abs(rho_L) >= 0.30 and p_L < 0.05) else 'NO_LIGHTNESS_LINK'
    say('P4 色相 n=%d  Mantel ρ=%+.4f p=%.4g (MDEρ=%.3f) => %s'
        % (len(chrom), rho_h, p_h, out['hue']['mde'], out['verdict']))
    say('P4 明度 ρ(L*,‖e‖)=%+.4f p=%.4f => %s' % (rho_L, p_L, out['verdict_L']))
    say('P4 单 token 稳健 n=%d ρ=%+.4f  上下文 L%d ρ=%+.4f'
        % (len(single), rho_h1, ctx.get('layer', -1), ctx.get('rho', float('nan'))))
    return out


# ============================ P5 生僻字 ============================

def p5_rare(E, mu, sets, tok, types, dec):
    words = sets['RARE']['words']
    ids_a = sets['RARE']['ids_a']
    ids_b = sets['RARE']['ids_b']
    n_multi = sum(1 for x in ids_a if x is None)
    cov = n_multi / len(words)

    bye = [w for w, x in zip(words, ids_a) if x is None]
    single = [w for w, x in zip(words, ids_a) if x is not None]

    # 全码位普查：哪些 CJK 字不在词表里（字节回退）
    scan = []
    if not SMOKE:
        for cp in range(0x4E00, 0x9FA5 + 1):
            ch = chr(cp)
            ids = tok.encode(ch, add_special_tokens=False)
            if len(ids) > 1:
                scan.append(cp)
    scan = np.array(scan, dtype=np.int64)

    def byte_vecs(chars):
        return np.array([E[tok.encode(c, add_special_tokens=False)].mean(0) for c in chars]) - mu

    out = dict(n_rare=len(words), n_multi_token=n_multi, coverage=float(cov),
               multi=bye, single=single,
               census_byte_fallback=int(len(scan)),
               census_range=20902)
    say('P5 覆盖：%d/%d 字节回退 (%.1f%%)；全 CJK 普查 0x4E00–0x9FA5 中 %d 字不在词表'
        % (n_multi, len(words), 100 * cov, len(scan)))

    # 码位律：在普查子集上抽 400 字，算 码位距离 ↔ cos
    if len(scan) > 10:
        pick = scan if len(scan) <= 400 else np.sort(RNG.choice(scan, 400, replace=False))
        chars = [chr(int(c)) for c in pick]
        Vb = byte_vecs(chars)
        cosv = cosine(Vb, Vb)
        Dcp = np.abs(pick[:, None] - pick[None, :]).astype(float)
        iu = np.triu_indices(len(pick), 1)
        rho_cp = spearman(Dcp[iu], cosv[iu])
        perm = np.array([spearman(Dcp[iu], cosv[np.ix_(p, p)][iu])
                         for p in (RNG.permutation(len(pick)) for _ in range(200 if not SMOKE else 40))])
        p_cp = float((np.abs(perm) >= abs(rho_cp)).mean())
        # 距离分箱看形状（D16）
        qs = np.quantile(Dcp[iu], np.linspace(0, 1, 9))
        bins = []
        for i in range(8):
            m = (Dcp[iu] >= qs[i]) & (Dcp[iu] <= qs[i + 1])
            bins.append(dict(cp_lo=float(qs[i]), cp_hi=float(qs[i + 1]),
                             cos_mean=float(cosv[iu][m].mean()), n=int(m.sum())))
        n_pair = int(iu[0].size)
        out['codepoint'] = dict(n=len(pick), n_pairs=n_pair, rho=rho_cp, p=p_cp, bins=bins,
                               mde=mde_spearman(n_pair))
        say('P5 码位 n=%d 对 %d  ρ_s(码位距离, cos)=%+.4f p=%.4g (MDE %.3f)'
            % (len(pick), n_pair, rho_cp, p_cp, mde_spearman(n_pair))) 
        say('P5 码位八段 cos: ' + ' '.join('%.3f' % b['cos_mean'] for b in bins))

    # 塌缩：within(RARE) vs within(COMMON)
    Vr = np.array([E[i].mean(0) for i in ids_b]) - mu
    Vc = np.array([E[i] for i in sets['COMMON']['ids_a'] if i is not None]) - mu
    w_r, w_c = within_mean(Vr), within_mean(Vc)
    nulls = []
    for _ in range(N_PERM):
        idx = RNG.choice(E.shape[0], len(Vc), replace=False)
        nulls.append(within_mean(E[idx] - mu))
    nulls = np.array(nulls)
    idmap = dict(zip(words, ids_a))
    Vbye = np.array([E[i].mean(0) for w, i in zip(words, ids_b) if idmap[w] is None]) - mu
    Vsing = np.array([E[i] for i in ids_a if i is not None]) - mu
    w_bye = within_mean(Vbye) if len(Vbye) > 2 else float('nan')
    w_sing = within_mean(Vsing) if len(Vsing) > 2 else float('nan')
    out['subgroups'] = dict(n_byte=len(Vbye), n_single=len(Vsing),
                            within_byte=w_bye, within_single=w_sing)
    say('P5 子群 within：字节回退(n=%d) %.4f  vs  在表单字(n=%d) %.4f'
        % (len(Vbye), w_bye, len(Vsing), w_sing))
    out['collapse'] = dict(within_rare=w_r, within_common=w_c, diff=float(w_r - w_c),
                           rand_null_mean=float(nulls.mean()),
                           p=float((nulls >= w_r).mean()))
    say('P5 塌缩 within(RARE)=%.4f within(COMMON)=%.4f Δ=%+.4f (随机 %.4f) => %s'
        % (w_r, w_c, w_r - w_c, nulls.mean(),
           'PASS' if (w_r - w_c) >= 0.10 else 'FAIL'))

    # 邻居构成（预注册 (d)）：混合池 = 字节回退字 + 常见在表汉字，看近邻是否"只认同类"
    all_chars = [chr(int(c)) for c in scan] if len(scan) else []
    if len(all_chars) > 20:
        rare_pool = all_chars if len(all_chars) <= 400 else [
            all_chars[k] for k in RNG.choice(len(all_chars), 400, replace=False)]
        com_idx = [i for i in np.where(types == 'char')[0]
                   if dec[i] not in set(rare_pool)][:6000]
        com_pool = [dec[i] for i in RNG.choice(com_idx, min(800, len(com_idx)), replace=False)]
        pool_chars = rare_pool + com_pool
        is_rare = np.array([1] * len(rare_pool) + [0] * len(com_pool))
        Vp = byte_vecs(pool_chars)
        G = l2n(Vp)
        S = G @ G.T
        np.fill_diagonal(S, -9)
        tgt = RNG.choice(len(rare_pool), min(50, len(rare_pool)), replace=False)
        nb = np.argsort(-S[tgt], axis=1)[:, :10]
        share = float(np.mean(is_rare[nb]))
        base = float(is_rare.mean())
        out['nn'] = dict(n_target=len(tgt), k=10, share_rare_neighbors=share,
                         base_rate=base, enrichment=share / base if base else float('nan'))
        say('P5 邻居构成：生僻字 10 近邻中生僻字占 %.2f（池基线 %.2f，富集 %.2f×）'
            % (share, base, share / base if base else float('nan')))

    # 判据汇总
    cond = dict(cov=cov >= 0.30, codepoint=out.get('codepoint', {}).get('rho', -1) >= 0.50,
                collapse=(w_r - w_c) >= 0.10,
                nn=out.get('nn', {}).get('share_rare_neighbors', -1) >= 0.70)
    out['cond'] = {k: bool(v) for k, v in cond.items()}
    npass = sum([out['cond']['codepoint'], out['cond']['collapse'], out['cond']['nn']])
    out['verdict'] = ('BYTE_COMPOSED_STRONG' if npass == 3 else
                      ('BYTE_COMPOSED' if npass >= 2 else 'NOT_BYTE_COMPOSED'))
    say('P5 判决 => %s  (%s)' % (out['verdict'], out['cond']))
    return out


# ============================ P6 形旁 ============================

def p6_radical(E, mu, sets, char_pool):
    groups = {}
    for k in RADICAL:
        name = 'RAD' + k
        ids = [x for x in sets[name]['ids_a'] if x is not None]
        groups[k] = np.array([E[i] for i in ids]) - mu
    # 全局 between：把 8 组拼起来算 between 均值
    allX = np.vstack(list(groups.values()))
    n = len(allX)
    G = l2n(allX)
    sizes = [len(v) for v in groups.values()]
    idx = np.cumsum([0] + sizes)
    tot = 0.0
    cnt = 0
    for i in range(len(sizes)):
        for j in range(i + 1, len(sizes)):
            blk = G[idx[i]:idx[i + 1]] @ G[idx[j]:idx[j + 1]].T
            tot += blk.sum()
            cnt += blk.size
    between = tot / cnt

    res = {}
    for k, X in groups.items():
        w = within_mean(X)
        d = w - between
        nulls = []
        nk = len(X)
        for _ in range(N_PERM):
            pick = RNG.choice(char_pool, nk, replace=False)
            Y = E[pick] - mu
            nulls.append(within_mean(Y))
        nulls = np.array(nulls)
        z = (w - nulls.mean()) / max(nulls.std(), 1e-9)
        res[k] = dict(n=nk, within=w, diff=d, null_mean=float(nulls.mean()),
                      null_p95=float(np.percentile(nulls, 95)), z=float(z),
                      p=float((nulls >= w).mean()),
                      homog=(k in RAD_HOMOG), hetero=(k in RAD_HETERO))
    hom = [v['diff'] for v in res.values() if v['homog']]
    het = [v['diff'] for v in res.values() if v['hetero']]
    out = dict(groups=res, between=float(between),
               homog_mean=float(np.mean(hom)), hetero_mean=float(np.mean(het)),
               ratio=float(np.mean(het) / np.mean(hom)) if np.mean(hom) else float('nan'))
    if out['ratio'] >= 0.5:
        out['verdict'] = 'ORTHOGRAPHIC_CONTRIBUTES'
    elif out['hetero_mean'] <= 0.01:
        out['verdict'] = 'SEMANTIC_ONLY'
    else:
        out['verdict'] = 'PARTIAL'
    for k, v in res.items():
        say('P6 %-2s n=%2d within=%.4f Δ=%.4f (null %.4f±) z=%+.1f p=%.4f %s'
            % (k, v['n'], v['within'], v['diff'], v['null_mean'], v['z'], v['p'],
               '同质' if v['homog'] else ('异质' if v['hetero'] else '')))
    say('P6 同质组 Δ 均 %+.4f  异质组 Δ 均 %+.4f  比值 %.2f => %s'
        % (out['homog_mean'], out['hetero_mean'], out['ratio'], out['verdict']))
    return out


# ============================ P7 单坐标载荷 ============================

def eta2_matrix(X, labels):
    """X: (n,d) 原始坐标；labels: (n,) int。返回每个坐标的 η²。"""
    n, d = X.shape
    tot = X - X.mean(0)
    ss_tot = (tot ** 2).sum(0)
    ss_b = np.zeros(d)
    for g in np.unique(labels):
        m = labels == g
        ss_b += m.sum() * (X[m].mean(0) - X.mean(0)) ** 2
    return ss_b / np.maximum(ss_tot, 1e-12)


def p7_coords(E, mu, sets):
    """坐标普查 + top-k 留出曲线 + 频率混淆的双重排除。

    跑后发现的关键陷阱（§8）：4 组在 zipf 上差 4 个数量级，top-1 坐标与 zipf 的 ρ 高达 0.80，
    原始 top-k 曲线测的是"频率轴"而不是"类别轴"。因此并列报两套曲线：
      raw     原始坐标
      resid   每个坐标对 [1, zipf, 字节长, log‖e‖] 回归后的残差坐标（去掉可被三个标量解释的部分）
    判决用 resid。
    """
    from wordfreq import zipf_frequency as zf
    names = ['COLOR', 'FRUIT', 'RARE', 'COMMON']
    Xs, ys, allw = [], [], []
    for gi, nm in enumerate(names):
        ids = sets[nm]['ids_b']
        Xs.append(np.array([E[i].mean(0) for i in ids]))
        ys += [gi] * len(ids)
        allw += sets[nm]['words']
    X = np.vstack(Xs)
    y = np.array(ys)
    z = np.array([zf(w, 'zh') for w in allw])
    blen = np.array([len(w.encode('utf-8')) for w in allw], dtype=float)
    lnorm = np.log(np.linalg.norm(X, axis=1))
    D = np.column_stack([np.ones(len(X)), z, blen, lnorm])
    beta, *_ = np.linalg.lstsq(D, X, rcond=None)
    R = X - D @ beta

    eth = eta2_matrix(X, y)
    eth_r = eta2_matrix(R, y)

    # 单个坐标与 zipf 的相关谱：频率是否是"一个坐标就能读"的量
    zc = (z - z.mean()) / z.std()
    Xc = (X - X.mean(0)) / X.std(0)
    rho_f = (Xc * zc[:, None]).mean(0)
    j_f = int(np.argmax(np.abs(rho_f)))
    say('P7 频率轴：单坐标 |ρ(坐标, zipf)| 最大 = %.3f（坐标 %d）；前五 %s'
        % (rho_f[j_f], j_f, np.round(np.sort(-np.abs(rho_f))[:5] * -1, 3)))

    def perm_max_curve(eth_obs, Xm, n_perm):
        pm = np.empty(n_perm)
        for t in range(n_perm):
            pm[t] = eta2_matrix(Xm, RNG.permutation(y)).max()
        return float(pm.max()), float(pm.mean())

    thr, pm_mean = perm_max_curve(eth, X, N_PERM_COORD)
    thr_r, pm_mean_r = perm_max_curve(eth_r, R, N_PERM_COORD)
    n_coord = int((eth >= thr).sum())
    n_coord_r = int((eth_r >= thr_r).sum())
    say('P7 坐标普查 d=%d  raw max η²=%.4f（perm-max %.4f, 均值 %.4f）⇒ 超阈 %d 个'
        % (X.shape[1], eth.max(), thr, pm_mean, n_coord))
    say('P7 残差坐标 max η²=%.4f（perm-max %.4f, 均值 %.4f）⇒ 超阈 %d 个'
        % (eth_r.max(), thr_r, pm_mean_r, n_coord_r))

    diag = []
    for j in np.argsort(-eth)[:5]:
        col = X[:, int(j)]
        diag.append(dict(coord=int(j), eta2=float(eth[j]),
                         group_means=[float(col[y == g].mean()) for g in range(4)],
                         rho_zipf=spearman(col, z), rho_bytelen=spearman(col, blen)))
        say('P7 top坐标 %5d η²=%.4f 组均 ' % (j, eth[j]) +
            ' '.join('%.3f' % v for v in diag[-1]['group_means']) +
            '  ρ(zipf)=%+.3f ρ(字节长)=%+.3f'
            % (diag[-1]['rho_zipf'], diag[-1]['rho_bytelen']))

    def classify(Xtr, ytr, Xte, yte, k):
        e = eta2_matrix(Xtr, ytr)
        sel = np.argsort(-e)[:k]
        cents = np.array([Xtr[ytr == g][:, sel].mean(0) for g in range(4)])
        d = ((Xte[:, sel][:, None, :] - cents[None]) ** 2).sum(-1)
        return float((d.argmin(1) == yte).mean())

    idx = RNG.permutation(len(X))
    halves = [idx[:len(X) // 2], idx[len(X) // 2:]]
    ks = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, X.shape[1]]
    curves = {}
    for tag, Xm in (('raw', X), ('resid', R)):
        c = []
        for k in ks:
            accs = []
            for a, b in ((0, 1), (1, 0)):
                tr, te = halves[a], halves[b]
                accs.append(classify(Xm[tr], y[tr], Xm[te], y[te], k))
            c.append(dict(k=int(k), acc=float(np.mean(accs)), accs=[float(v) for v in accs]))
        curves[tag] = c
        say('P7 top-k 留出曲线[%s]（机会 0.25）：' % tag +
            ' '.join('k=%d:%.3f' % (x['k'], x['acc']) for x in c))

    acc32 = [x['acc'] for x in curves['raw'] if x['k'] == 32][0]
    acc32r = [x['acc'] for x in curves['resid'] if x['k'] == 32][0]

    # 频率分层零假设：只在 zipf 十分位内打乱标签，任何"频率可解释"的 η² 被吸收
    dec = np.digitize(z, np.quantile(z, np.linspace(0, 1, 11))[1:-1])
    e_strat = []
    for _ in range(N_PERM_COORD):
        yp = y.copy()
        for d in np.unique(dec):
            m = dec == d
            yp[m] = RNG.permutation(y[m])
        e_strat.append(eta2_matrix(X, yp).max())
    thr_s = float(np.max(e_strat))
    n_strat = int((eth >= thr_s).sum())
    say('P7 频率分层零假设 max η² 上界 = %.4f ⇒ 超阈坐标 %d 个（对照 raw %d）'
        % (thr_s, n_strat, n_coord))

    # 主干重叠
    y_co = (y == 0).astype(int)
    y_fr = (y == 1).astype(int)
    top_co = set(np.argsort(-eta2_matrix(R, y_co))[:32].tolist())
    top_fr = set(np.argsort(-eta2_matrix(R, y_fr))[:32].tolist())
    ov = len(top_co & top_fr)
    exp = 32 * 32 / R.shape[1]
    p_ov = float(stats.hypergeom.sf(ov - 1, R.shape[1], 32, 32)) if ov > 0 else 1.0
    say('P7 主干重叠（残差坐标）COLOR∩FRUIT top-32 = %d（期望 %.2f，超几何 p=%.3g）=> %s'
        % (ov, exp, p_ov, 'SHARED_BACKBONE' if ov >= 8 else 'SEPARATE_COORDINATES'))

    out = dict(d=int(X.shape[1]), eta_max=float(eth.max()), perm_max=thr,
               perm_max_mean=pm_mean, n_coord_over_perm_max=n_coord,
               eta_max_resid=float(eth_r.max()), perm_max_resid=thr_r,
               n_coord_over_perm_max_resid=n_coord_r,
               freq_axis=dict(coord=j_f, rho_max=float(rho_f[j_f])),
               coord_diag=diag, curves=curves, acc_at_32=float(acc32),
               acc_at_32_resid=float(acc32r),
               freq_stratified=dict(thr=thr_s, n_coord=n_strat),
               overlap=dict(n=ov, expected=float(exp), p=p_ov),
               overlap_verdict='SHARED_BACKBONE' if ov >= 8 else 'SEPARATE_COORDINATES')
    if acc32r >= 0.75:
        out['verdict'] = 'PASS_ABOVE_FREQUENCY'
    elif acc32 >= 0.75:
        out['verdict'] = 'CONFOUNDED_BY_FREQUENCY'
    else:
        out['verdict'] = 'FAIL'
    say('P7 k=32 留出 raw %.3f / 去频率 %.3f => %s' % (acc32, acc32r, out['verdict']))
    return out


def p8_link(E, mu, sets, model, tok):
    import torch
    from wordfreq import zipf_frequency as zf
    names = ['FRUIT', 'COLOR']
    out = {}
    prompts = {'FRUIT': '我喜欢吃的水果有苹果、香蕉、', 'COLOR': '常见的颜色有红色、蓝色、'}
    for nm in names:
        ids = sets[nm]['ids_b']
        words = sets[nm]['words']
        V = np.array([E[i].mean(0) for i in ids]) - mu
        cent = V.mean(0)
        cs = cosine(V, cent[None])[:, 0]
        dev = model.get_input_embeddings().weight.device
        enc = tok(prompts[nm], return_tensors='pt').to(dev)
        with torch.no_grad():
            logits = model(**enc).logits[0, -1].float().cpu().numpy()
        lp = logits - np.log(np.exp(logits - logits.max()).sum()) - logits.max()
        first = np.array([tok.encode(w, add_special_tokens=False)[0] for w in words])
        y = lp[first]
        f = np.array([zf(w, 'zh') for w in words])
        r_p = partial_spearman(cs, y, f)
        perm = np.array([partial_spearman(cs, RNG.permutation(y), f)
                         for _ in range(1000 if not SMOKE else 200)])
        p = float((np.abs(perm) >= abs(r_p)).mean())
        single = [k for k in range(len(words)) if sets[nm]['ids_a'][k] is not None]
        r_s = (partial_spearman(cs[single], y[single], f[single]) if len(single) > 5
               else float('nan'))
        out[nm] = dict(n=len(words), rho_partial=float(r_p), p=p,
                       verdict=('PASS' if (r_p >= 0.30 and p < 0.05) else 'FAIL'),
                       rho_raw=spearman(cs, y), rho_freq=spearman(y, f),
                       single=dict(n=len(single), rho_partial=float(r_s)),
                       mde=mde_spearman(len(words)))
        say('P8 %s n=%d  偏ρ(cos,logprob|zipf)=%+.4f p=%.4f  原始ρ=%+.4f  ρ(logprob,zipf)=%+.4f  单token n=%d ρ=%+.4f => %s'
            % (nm, len(words), r_p, p, out[nm]['rho_raw'], out[nm]['rho_freq'],
               len(single), r_s, 'PASS' if (r_p >= 0.30 and p < 0.05) else 'FAIL'))
    return out


# ============================ main ============================

TOK = None


def _jsonable(o):
    """把 numpy 标量/数组与 NaN/Inf 换成 JSON 可序列化对象。"""
    if isinstance(o, dict):
        return {str(k): _jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_jsonable(v) for v in o]
    if isinstance(o, (np.floating, float)):
        v = float(o)
        return v if np.isfinite(v) else None
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, np.ndarray):
        return _jsonable(o.tolist())
    if isinstance(o, (np.bool_, bool)):
        return bool(o)
    return o


def main():
    global TOK
    t0 = time.time()
    say('=' * 78)
    say('Q8-A 静态嵌入解剖  model=%s  SMOKE=%s  seed=%d' % (MODEL_PATH, SMOKE, SEED))
    say('=' * 78)
    tok, model, E = load_model()
    TOK = tok
    say('E', E.shape, 'tied', getattr(model.config, 'tie_word_embeddings', None),
        'dtype', model.dtype)
    sets, dec, cjk = build_sets(tok, E)
    mu = E.mean(0)
    say('μ‖=%.4f  平均范数 %.4f' % (float(np.linalg.norm(mu)),
                                  float(np.linalg.norm(E, axis=1).mean())))
    for k, v in sets.items():
        if k in ('COLOR', 'FRUIT', 'RARE', 'COMMON'):
            say('set %-7s n=%2d  单 token=%2d' % (k, len(v['words']), v['n_single']))

    payload = dict(config=dict(model=MODEL_PATH, smoke=SMOKE, seed=SEED,
                               n_perm=N_PERM, n_perm_mantel=N_PERM_MANTEL,
                               n_perm_coord=N_PERM_COORD,
                               mu_norm=float(np.linalg.norm(mu)),
                               mean_norm=float(np.linalg.norm(E, axis=1).mean())),
                   sets={k: dict(n=len(v['words']), n_single=v['n_single'], words=v['words'])
                         for k, v in sets.items()})

    payload['P0'] = p0_neighbors(E, mu, dec, sets,
                                 ['红', '苹果', '江', '破', '龘', '的', '爱', '三'])
    payload['P9'] = p9_translation(E, mu, dec, tok)
    payload['P10'] = p10_simp_trad(E, mu, dec, tok)
    p1, types = p1_unit_types(E, dec, cjk)
    payload['P1'] = p1
    payload['P2'] = p2_freq_norm(E, dec, cjk, types)
    pool = freq_matched_pool(dec, cjk, types)
    say('频率匹配池 n=%d（双字纯汉字 token）' % len(pool))
    payload['P3'] = p3_category(E, mu, sets, pool)
    payload['P4'] = p4_hue(E, mu, sets, model, tok)
    payload['P5'] = p5_rare(E, mu, sets, tok, types, dec)
    rad_used = set(''.join(RADICAL.values()))
    char_pool = np.array([i for i in np.where(types == 'char')[0] if dec[i] not in rad_used])
    say('形旁零分布池 n=%d（在表单字，剔除 8 组本身）' % len(char_pool))
    payload['P6'] = p6_radical(E, mu, sets, char_pool)
    payload['P7'] = p7_coords(E, mu, sets)
    try:
        payload['P8'] = p8_link(E, mu, sets, model, tok)
    except Exception as e:
        payload['P8'] = dict(error=repr(e))
        say('P8 失败：%r' % (e,))

    # 判决汇总
    verd = {}
    for k in ('P0', 'P1', 'P2', 'P3', 'P4', 'P5', 'P6', 'P7', 'P8', 'P9', 'P10'):
        v = payload[k]
        cand = {kk: vv for kk, vv in v.items()
                if kk.startswith('verdict') and isinstance(vv, str)}
        verd[k] = cand
    payload['verdicts'] = verd
    say('-' * 78)
    for k, v in verd.items():
        say(k, v)
    say('耗时 %.1f s' % (time.time() - t0))
    payload['runtime_s'] = round(time.time() - t0, 1)

    path = provenance.write_result('pat_q8a', _jsonable(payload), model_path=MODEL_PATH,
                                  seed=SEED, tool='tools/pat_q8a.py')
    say('RESULT', path)
    txt = path.with_suffix('.log.txt')
    txt.write_text('\n'.join(LOG) + '\n', encoding='utf-8', newline='')
    say('LOG', txt)


if __name__ == '__main__':
    main()
