# -*- coding: utf-8 -*-
"""
E3b: 词嵌入特征审计 补充
修正 E3 的查表 bug（上位词未纳入），并加做三个决定性对照：
  (1) E1 关键对复现（raw/ctr + 随机 q95 标注）
  (2) 上位词是否落在"类质心"附近？ -> "is-a 不是几何距离"的直接检验
  (3) 类内紧致度 vs 类间距离；类质心的有效秩 vs 随机子集的有效秩
零 GPU，只读 embedding 表。
"""
import os, json, time
import numpy as np

ROOT = r'D:\AI2050\Ai2050-OpenOne'
OUT = os.path.join(ROOT, 'tests', 'gpt5_temp', 'e3b_report.txt')
lines = []
def w(s=''): lines.append(str(s))

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
    'furniture':['桌子','椅子','床','沙发','柜子','书架','板凳','地毯','窗帘','台灯'],
    'metal':   ['铁','铜','铝','金','银','锌','铅','锡','镍','钢'],
    'food':    ['米饭','面包','牛奶','鸡蛋','肉','面条','汤','粥','馒头','豆腐'],
    'celestial':['太阳','月亮','星星','地球','火星','木星','金星','土星'],
}
SUPERS = {'fruit':'水果','vehicle':'交通工具','animal':'动物','color':'颜色','emotion':'情绪',
          'number':'数字','furniture':'家具','metal':'金属','food':'食物','celestial':'天体'}
EXTRA = ['植物','公司','东西','物体','生物','机器']
PAIRS = [('苹果','水果'),('苹果','香蕉'),('苹果','梨'),('苹果','食物'),('苹果','植物'),
         ('香蕉','水果'),('香蕉','食物'),('水果','食物'),('食物','东西'),('苹果','公司')]

def load_embed(mdir):
    idx = os.path.join(mdir, 'model.safetensors.index.json'); key = f = None
    if os.path.exists(idx):
        J = json.load(open(idx, encoding='utf-8'))
        for k, v in J['weight_map'].items():
            if k.endswith('embed_tokens.weight'): key, f = k, v; break
    if key is None:
        for fn in sorted(os.listdir(mdir)):
            if fn.endswith('.safetensors'):
                with safe_open(os.path.join(mdir, fn), framework='pt') as fo:
                    for k in fo.keys():
                        if k.endswith('embed_tokens.weight'): key, f = k, fn; break
            if key: break
    with safe_open(os.path.join(mdir, f), framework='pt') as fo:
        return fo.get_tensor(key).float().numpy()

def cos(a, b):
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    return 0.0 if na < 1e-9 or nb < 1e-9 else float(np.dot(a, b) / (na * nb))

def pr(X):
    Xc = X - X.mean(0, keepdims=True)
    s2 = np.linalg.svd(Xc, compute_uv=False) ** 2
    return float((s2.sum() ** 2) / (s2 ** 2).sum()) if s2.sum() > 0 else 0.0

rng = np.random.default_rng(1)
for name, mdir in MODELS:
    w('=' * 74); w('MODEL %s' % name); w('=' * 74)
    E = load_embed(mdir); n, H = E.shape
    tok = AutoTokenizer.from_pretrained(mdir, trust_remote_code=True)
    samp = rng.choice(n, size=min(20000, n), replace=False)
    mu = E[samp].mean(0); Ec = E - mu
    w('  shape=(%d,%d) ||mu||/mean_row_norm=%.4f/%.4f' % (n, H, np.linalg.norm(mu), np.mean(np.linalg.norm(E[samp], axis=1))))

    idmap = {}
    for g, ws in GROUPS.items():
        for x in ws:
            t = tok.encode(x, add_special_tokens=False)
            if len(t) == 1: idmap[x] = t[0]
    for x in list(SUPERS.values()) + EXTRA:
        t = tok.encode(x, add_special_tokens=False)
        if len(t) == 1: idmap[x] = t[0]

    ra = rng.choice(n, 6000, replace=False); rb = rng.choice(n, 6000, replace=False)
    rn = np.array([abs(cos(E[a], E[b])) for a, b in zip(ra, rb)])
    cn = np.array([abs(cos(Ec[a], Ec[b])) for a, b in zip(ra, rb)])
    w('  NULL q95: raw=%.4f ctr=%.4f' % (np.quantile(rn, .95), np.quantile(cn, .95)))

    w('  -- (1) E1 关键对 [raw | ctr]  (^=超过 raw q95, *=超过 ctr q95) --')
    for a, b in PAIRS:
        if a in idmap and b in idmap:
            r = cos(E[idmap[a]], E[idmap[b]]); c = cos(Ec[idmap[a]], Ec[idmap[b]])
            w('     %-4s-%-6s raw=%+.4f%s  ctr=%+.4f%s' % (
                a, b, r, '^' if abs(r) > np.quantile(rn, .95) else ' ',
                c, '*' if abs(c) > np.quantile(cn, .95) else ' '))
        else:
            w('     %-4s-%-6s (缺 id)' % (a, b))

    w('  -- (2) 上位词 vs 类质心：cos(super, centroid) 对上 cos(instance, centroid) --')
    rnd_ids = rng.choice(n, 400, replace=False)
    for g, ws in GROUPS.items():
        sup = SUPERS[g]
        mem = [x for x in ws if x in idmap]
        if sup not in idmap or len(mem) < 4: continue
        M = np.array([E[idmap[x]] for x in mem]); Mc = np.array([Ec[idmap[x]] for x in mem])
        ctr = M.mean(0); ctrc = Mc.mean(0)
        ci = np.mean([cos(E[idmap[x]], ctr) for x in mem])
        cs = cos(E[idmap[sup]], ctr)
        cr = np.mean([cos(E[i], ctr) for i in rnd_ids])
        w('     %-10s sup=%-5s cos(mem,ctr)=%.4f  cos(SUP,ctr)=%.4f  cos(rand,ctr)=%.4f  ->  SUP%s mem' %
          (g, sup, ci, cs, cr, '>' if cs > ci else '<'))
    w('  -- (2b) 同上，centered --')
    for g, ws in GROUPS.items():
        sup = SUPERS[g]
        mem = [x for x in ws if x in idmap]
        if sup not in idmap or len(mem) < 4: continue
        Mc = np.array([Ec[idmap[x]] for x in mem]); ctrc = Mc.mean(0)
        ci = np.mean([cos(Ec[idmap[x]], ctrc) for x in mem])
        cs = cos(Ec[idmap[sup]], ctrc)
        cr = np.mean([cos(Ec[i], ctrc) for i in rnd_ids])
        w('     %-10s sup=%-5s cos(mem,ctr)=%.4f  cos(SUP,ctr)=%.4f  cos(rand,ctr)=%.4f' % (g, sup, ci, cs, cr))

    # (3) 类内紧致度 vs 类间
    def compact(tag, mat):
        inw, outw = [], []
        gs = [g for g, ws in GROUPS.items() if len([x for x in ws if x in idmap]) >= 4]
        for g in gs:
            mem = [x for x in GROUPS[g] if x in idmap]
            for i in range(len(mem)):
                for j in range(i + 1, len(mem)):
                    inw.append(cos(mat[idmap[mem[i]]], mat[idmap[mem[j]]]))
        for gi in range(len(gs)):
            for gj in range(gi + 1, len(gs)):
                A = [x for x in GROUPS[gs[gi]] if x in idmap]; B = [x for x in GROUPS[gs[gj]] if x in idmap]
                for a in A:
                    for b in B:
                        outw.append(cos(mat[idmap[a]], mat[idmap[b]]))
        w('     %s within-class mean=%+.4f (n=%d) | between-class mean=%+.4f (n=%d) | gap=%+.4f' %
          (tag, np.mean(inw), len(inw), np.mean(outw), len(outw), np.mean(inw) - np.mean(outw)))
    w('  -- (3) 类内/类间 cos --'); compact('RAW', E); compact('CTR', Ec)

    # (4) 类质心有效秩 vs 随机
    cents = []
    for g, ws in GROUPS.items():
        mem = [x for x in ws if x in idmap]
        if len(mem) >= 4: cents.append(E[[idmap[x] for x in mem]].mean(0))
    w('  -- (4) PR: class centroids=%.2f (k=%d) | random 400 rows=%.2f | random %d rows=%.2f' %
      (pr(np.array(cents)), len(cents), pr(E[rng.choice(n, 400, replace=False)]),
       len(cents), pr(E[rng.choice(n, len(cents), replace=False)])))

open(OUT, 'w', encoding='utf-8').write('\n'.join(lines))
print('DONE')
