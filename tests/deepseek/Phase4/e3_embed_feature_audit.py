# -*- coding: utf-8 -*-
"""
E3: 词嵌入特征审计（raw vs centered）
目的：检验"词嵌入必然带有某种特征"这一命题；并修正 E1 的方法缺陷
      —— E1 使用未居中余弦，可能被公共模（common-mode）分量污染。
只读 embedding 表，不加载整模型，零 GPU。
"""
import os, json, time, math, random, sys
import numpy as np

ROOT = r'D:\AI2050\Ai2050-OpenOne'
OUT = os.path.join(ROOT, 'tests', 'gpt5_temp', 'e3_report.txt')
lines = []
def w(s=''):
    lines.append(str(s))

from safetensors import safe_open
from transformers import AutoTokenizer

MODELS = [
    ('qwen3-4b', os.path.join(ROOT, 'models', 'hf', 'qwen3-4b')),
    ('glm4-9b-chat-hf', os.path.join(ROOT, 'models', 'hf', 'glm4-9b-chat-hf')),
    ('qwen2.5-3b-instruct', os.path.join(ROOT, 'models', 'hf', 'qwen2.5-3b-instruct')),
]

GROUPS = {
    'fruit':   ['苹果','香蕉','梨','桃子','葡萄','西瓜','草莓','橙子','芒果','柠檬','樱桃','菠萝','橘子','荔枝'],
    'vehicle': ['汽车','火车','飞机','轮船','自行车','摩托车','公交车','出租车','卡车','地铁'],
    'animal':  ['狗','猫','老虎','大象','兔子','猴子','马','牛','羊','鸟','鱼','猪','鸡','老鼠'],
    'color':   ['红','蓝','绿','黄','黑','白','紫','灰','粉','棕'],
    'emotion': ['高兴','悲伤','愤怒','害怕','惊讶','讨厌','焦虑','兴奋','孤单','满足'],
    'number':  ['一','二','三','四','五','六','七','八','九','十','百','千'],
    'furniture': ['桌子','椅子','床','沙发','柜子','书架','板凳','地毯','窗帘','台灯'],
    'metal':   ['铁','铜','铝','金','银','锌','铅','锡','镍','钢'],
    'food':    ['米饭','面包','牛奶','鸡蛋','肉','面条','汤','粥','馒头','豆腐'],
    'celestial': ['太阳','月亮','星星','地球','火星','木星','金星','土星'],
}
SUPERS = {
    'fruit':'水果','vehicle':'交通工具','animal':'动物','color':'颜色',
    'emotion':'情绪','number':'数字','furniture':'家具','metal':'金属',
    'food':'食物','celestial':'天体',
}
# 层次链（is-a 链）：实例 -> 上位1 -> 上位2
CHAINS = [
    ('苹果','水果','食物'), ('香蕉','水果','食物'), ('梨','水果','食物'),
    ('狗','动物','生物'),   ('猫','动物','生物'),
    ('汽车','交通工具','机器'),
    ('米饭','食物','东西'),
]

def load_embed(mdir):
    idx = os.path.join(mdir, 'model.safetensors.index.json')
    key = None; f = None
    if os.path.exists(idx):
        J = json.load(open(idx, encoding='utf-8'))
        for k, v in J['weight_map'].items():
            if k.endswith('embed_tokens.weight'):
                key, f = k, v; break
    if key is None:
        for fn in sorted(os.listdir(mdir)):
            if fn.endswith('.safetensors'):
                with safe_open(os.path.join(mdir, fn), framework='pt') as fo:
                    for k in fo.keys():
                        if k.endswith('embed_tokens.weight'):
                            key, f = k, fn; break
            if key: break
    if key is None:
        raise RuntimeError('embed key not found in ' + mdir)
    with safe_open(os.path.join(mdir, f), framework='pt') as fo:
        t = fo.get_tensor(key)
    return t.float().numpy(), key

def cos(a, b):
    na = np.linalg.norm(a); nb = np.linalg.norm(b)
    if na < 1e-9 or nb < 1e-9: return 0.0
    return float(np.dot(a, b) / (na * nb))

def participation_ratio(X):
    Xc = X - X.mean(0, keepdims=True)
    s = np.linalg.svd(Xc, compute_uv=False)
    s2 = s ** 2
    if s2.sum() <= 0: return 0.0
    return float((s2.sum() ** 2) / (s2 ** 2).sum())

rng = np.random.default_rng(0)

for name, mdir in MODELS:
    w('=' * 72)
    w('MODEL %s' % name)
    w('=' * 72)
    t0 = time.time()
    try:
        E, key = load_embed(mdir)
    except Exception as e:
        w('  LOAD FAIL %r' % e); continue
    tok = AutoTokenizer.from_pretrained(mdir, trust_remote_code=True)
    w('  embed key=%s shape=%s dtype=float32 load=%.1fs' % (key, E.shape, time.time() - t0))

    # ---- 公共模（centering 参考）----
    n = E.shape[0]
    samp = rng.choice(n, size=min(20000, n), replace=False)
    mu = E[samp].mean(0)
    norm_mu = float(np.linalg.norm(mu))
    mean_norm = float(np.mean(np.linalg.norm(E[samp], axis=1)))
    w('  ||mu||=%.4f  mean_row_norm=%.4f  ratio=%.4f' % (norm_mu, mean_norm, norm_mu / mean_norm))
    Ec = E - mu

    # ---- 单 token 过滤 ----
    ids = {}
    dropped = []
    for g, ws in GROUPS.items():
        for wd in ws:
            t = tok.encode(wd, add_special_tokens=False)
            if len(t) == 1: ids[wd] = t[0]
            else: dropped.append(wd)
    sup_ids = {}
    for g, wd in SUPERS.items():
        t = tok.encode(wd, add_special_tokens=False)
        if len(t) == 1: sup_ids[wd] = t[0]
        else: dropped.append(wd)
    w('  single-token concepts=%d dropped=%d' % (len(ids), len(dropped)))
    w('  dropped: %s' % ' '.join(dropped[:40]))

    # ---- 随机零假设带 ----
    ra = rng.choice(n, size=6000, replace=False)
    rb = rng.choice(n, size=6000, replace=False)
    raw_null = np.array([abs(cos(E[a], E[b])) for a, b in zip(ra, rb)])
    ctr_null = np.array([abs(cos(Ec[a], Ec[b])) for a, b in zip(ra, rb)])
    w('  NULL |cos| raw: mean=%.4f q50=%.4f q95=%.4f q99=%.4f' %
      (raw_null.mean(), np.quantile(raw_null, .5), np.quantile(raw_null, .95), np.quantile(raw_null, .99)))
    w('  NULL |cos| ctr: mean=%.4f q50=%.4f q95=%.4f q99=%.4f' %
      (ctr_null.mean(), np.quantile(ctr_null, .5), np.quantile(ctr_null, .95), np.quantile(ctr_null, .99)))

    # ---- 命名对 ----
    def show(pairs, title):
        w('  -- %s (raw / centered) --' % title)
        for a, b in pairs:
            if a in ids and b in ids:
                w('     %-6s %-8s raw=%+.4f  ctr=%+.4f' % (a, b, cos(E[ids[a]], E[ids[b]]), cos(Ec[ids[a]], Ec[ids[b]])))

    show([('苹果','水果'),('苹果','香蕉'),('苹果','食物'),('苹果','植物'),('苹果','梨'),
          ('香蕉','水果'),('香蕉','食物'),('水果','食物'),('食物','东西') if False else ('水果','食物')],
         'E1 关键对')
    show([('苹果','公司'),('苹果','小米'),('病毒','细菌') if False else ('苹果','香蕉')], '词义/同形')

    # ---- 层次聚合：inst->super  vs  inst->co-inst ----
    def hierarchy(tag, mat):
        sup_v, co_v = [], []
        for g, ws in GROUPS.items():
            sup = SUPERS[g]
            if sup not in sup_ids: continue
            mem = [wd for wd in ws if wd in ids]
            if len(mem) < 3: continue
            for i, a in enumerate(mem):
                sup_v.append(cos(mat[ids[a]], mat[sup_ids[sup]]))
                for b in mem:
                    if b != a:
                        co_v.append(cos(mat[ids[a]], mat[ids[b]]))
        sup_v = np.array(sup_v); co_v = np.array(co_v)
        w('  %s: inst->super mean=%+.4f (n=%d) | inst->co-inst mean=%+.4f (n=%d) | delta=%+.4f' %
          (tag, sup_v.mean(), len(sup_v), co_v.mean(), len(co_v), sup_v.mean() - co_v.mean()))
        return sup_v.mean(), co_v.mean()
    hierarchy('RAW ', E)
    hierarchy('CTR ', Ec)

    # ---- is-a 链梯度 ----
    w('  -- is-a 链: cos(inst,sup1) > cos(sup1,sup2) ? --')
    for a, s1, s2 in CHAINS:
        if all(x in ids for x in (a, s1, s2)):
            w('     %s-%s raw=%+.4f ctr=%+.4f | %s-%s raw=%+.4f ctr=%+.4f' %
              (a, s1, cos(E[ids[a]],E[ids[s1]]), cos(Ec[ids[a]],Ec[ids[s1]]),
               s1, s2, cos(E[ids[s1]],E[ids[s2]]), cos(Ec[ids[s1]],Ec[ids[s2]])))

    # ---- 类别探针（leave-one-out 最近质心）----
    def probe(tag, mat):
        allwd = []; labels = []
        gs = [g for g, ws in GROUPS.items()
              if SUPERS[g] in sup_ids and len([x for x in ws if x in ids]) >= 4]
        for gi, g in enumerate(gs):
            for wd in GROUPS[g]:
                if wd in ids:
                    allwd.append(wd); labels.append(gi)
        X = np.array([mat[ids[x]] for x in allwd])
        y = np.array(labels)
        wrong = 0
        for i in range(len(allwd)):
            m = np.ones(len(allwd), bool); m[i] = False
            cents = []
            for gi in range(len(gs)):
                idx = np.where((y == gi) & m)[0]
                if len(idx) == 0: cents.append(np.zeros(X.shape[1])); continue
                cents.append(X[idx].mean(0))
            C = np.array(cents)
            Xn = X[i] / (np.linalg.norm(X[i]) + 1e-9)
            Cn = C / (np.linalg.norm(C, axis=1, keepdims=True) + 1e-9)
            pred = int(np.argmax(Cn @ Xn))
            if pred != y[i]: wrong += 1
        w('  PROBE %s: classes=%d items=%d LOO-acc=%.4f (chance=%.4f)' %
          (tag, len(gs), len(allwd), 1 - wrong / len(allwd), 1 / len(gs)))
        return X, y, gs
    probe('RAW ', E)
    probe('CTR ', Ec)

    # ---- 有效秩（类别质心 / 全样本子集）----
    w('  PR(class centroids, raw)=%.2f' % participation_ratio(E[samp]))
    w('  PR(class centroids, ctr)=%.2f' % participation_ratio(Ec[samp]))

    sys.stdout.flush()

open(OUT, 'w', encoding='utf-8').write('\n'.join(lines))
print('DONE ->', OUT)
