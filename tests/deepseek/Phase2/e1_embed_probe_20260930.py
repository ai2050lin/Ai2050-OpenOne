# -*- coding: utf-8 -*-
"""
E1 探索性探针：词嵌入层面的"知识是否住在几何里"
模型：qwen3-4b（零 GPU，仅读 embed_tokens 行）
协议对齐 Phase 2811/2812/2813（零前向，z 归一化 + 类零和对比方向 unitD + 能量份额分解）
状态：探索性（非预注册 Phase）。已在册结论（2811/2814/2815）先验指向"否证"，
      本探针目的 = 用中文通俗概念（苹果/香蕉/水果/食物/物体/公司）给出可直接阅读的数字。
"""
import os, json, math, time
import numpy as np

ROOT = r'D:\AI2050\Ai2050-OpenOne'
MODEL = os.path.join(ROOT, 'models', 'hf', 'qwen3-4b')
OUT = os.path.join(ROOT, 'tests', 'gpt5_temp', 'e1_embed_probe_report.txt')

t0 = time.time()
L = []
def w(s=''):
    L.append(str(s))

CATS = {
    'fruit':     ['苹果', '香蕉', '梨', '橘子', '葡萄', '桃子', '草莓', '西瓜'],
    'vegetable': ['白菜', '萝卜', '菠菜', '土豆', '黄瓜', '茄子'],
    'animal':    ['狗', '猫', '马', '牛', '老虎', '兔子', '猪', '羊'],
    'metal':     ['铁', '铜', '锌', '铝', '铅', '镍', '锡'],
    'country':   ['中国', '美国', '日本', '法国', '德国', '印度', '英国', '巴西'],
    'food':      ['食物', '食品', '饭菜', '米饭', '面包', '蛋糕', '面条'],
    'plant':     ['植物', '树', '花', '草', '叶子', '根'],
    'object':    ['物体', '东西', '物品', '工具', '装置', '器具'],
    'company':   ['公司', '企业', '集团', '品牌', '厂商', '机构'],
    'vehicle':   ['汽车', '飞机', '火车', '轮船', '自行车', '卡车'],
}
# 用户点名的比较对象
PROBES = ['苹果', '香蕉', '梨', '橘子', '水果', '食物', '物体', '植物', '动物', '公司']
# 用户点名的关系对
PAIRS = [('苹果', '香蕉'), ('苹果', '梨'), ('苹果', '水果'), ('苹果', '食物'),
         ('苹果', '植物'), ('苹果', '物体'), ('苹果', '公司'),
         ('香蕉', '水果'), ('水果', '食物'), ('食物', '物体'),
         ('水果', '植物'), ('植物', '物体'), ('狗', '动物'), ('狗', '猫'), ('铁', '铜')]

# ---------- tokenizer ----------
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained(MODEL)
def tids(s):
    return tok.encode(s, add_special_tokens=False)

w('=== E1 词嵌入探针（qwen3-4b, 零 GPU）===')
w('时间 %s' % time.strftime('%Y-%m-%d %H:%M:%S'))
w('模型 %s' % MODEL)
w('vocab_size(tokenizer) %d' % tok.vocab_size)

# ---------- 单 token 过滤 ----------
ok = {}
rej = {}
for c, ws in CATS.items():
    for x in ws:
        n = len(tids(x))
        if n == 1:
            ok[x] = tids(x)[0]
        else:
            rej[x] = n
for x in PROBES:
    if x not in ok and x not in rej:
        n = len(tids(x))
        (ok if n == 1 else rej)[x] = tids(x)[0] if n == 1 else n

w('')
w('[1] 单 token 存活：%d 个' % len(ok))
w('    存活: ' + ' '.join(sorted(ok.keys())))
w('    剔除(多 token): ' + (', '.join('%s->%dtok' % (k, v) for k, v in sorted(rej.items())) if rej else '无'))

# ---------- 读 embed_tokens ----------
from safetensors import safe_open
import torch
idx = json.load(open(os.path.join(MODEL, 'model.safetensors.index.json'), encoding='utf-8'))
wmap = idx['weight_map']
key = [k for k in wmap if k.endswith('embed_tokens.weight')][0]
shard = wmap[key]
w('')
w('[2] 权重: key=%s shard=%s' % (key, shard))
f = safe_open(os.path.join(MODEL, shard), framework='pt')
sl = f.get_slice(key)
D = sl.get_shape()[1]
V = sl.get_shape()[0]
w('    shape = (%d, %d)' % (V, D))

def rows(ids_arr):
    return np.stack([sl[int(i):int(i) + 1].to(torch.float32).numpy()[0] for i in ids_arr], 0)

# ---------- 随机基线 ----------
rng = np.random.default_rng(20260930)
RAND_N = 600
rand_ids = rng.choice(V, size=RAND_N, replace=False)
Erand = rows(rand_ids)

# ---------- z 归一化（2811 协议） ----------
def znorm(E):
    m = np.mean(E ** 2, axis=1, keepdims=True)
    return E / np.sqrt(m + 1e-6)

# 类别方向（2811 协议：Cm = 类均值行；dW = 零和对比空间；unitD = 单位化）
cm_names = [c for c in CATS if all(x in ok for x in CATS[c])]
cm_names = [c for c in cm_names if len([x for x in CATS[c] if x in ok]) >= 4]
Cm_rows = []
used_cat = {}
for c in cm_names:
    ws = [x for x in CATS[c] if x in ok]
    used_cat[c] = ws
    Cm_rows.append(np.mean(rows([ok[x] for x in ws]), axis=0))
Cm = np.stack(Cm_rows, 0)                    # (K, D)
K = Cm.shape[0]
dW = Cm - (Cm.sum(0, keepdims=True) - Cm) / (K - 1)   # 零和，rank K-1
unitD = dW / np.linalg.norm(dW, axis=1, keepdims=True)

w('')
w('[3] 类别方向空间: K=%d 类 -> unitD rank = %d' % (K, K - 1))
for i, c in enumerate(cm_names):
    w('    %-10s n=%d  %s' % (c, len(used_cat[c]), ' '.join(used_cat[c])))

# ---------- 类别投票谱 p(w) = z(w) . unitD^T ----------
probe_ok = [x for x in PROBES if x in ok]
Z = znorm(rows([ok[x] for x in probe_ok]))
P = Z @ unitD.T                                   # (n_probe, K)

w('')
w('[4] 类别投票谱 p(w)=z(w)·unitD^T  —— 回答"苹果的嵌入怎么表达出水果/食物/公司"')
w('    词        argmax类     top3(类:分) ')
for i, x in enumerate(probe_ok):
    o = np.argsort(-P[i])
    top3 = ', '.join('%s:%.3f' % (cm_names[j], P[i, j]) for j in o[:3])
    w('    %-8s  %-10s  %s' % (x, cm_names[o[0]], top3))

# ---------- 能量份额分解 ----------
Gall = znorm(np.concatenate([rows([ok[x] for x in probe_ok]), Erand], 0))
n_p = len(probe_ok)
Gp = Gall[:n_p]
Gr = Gall[n_p:]

def energy_split(G):
    # 公共方向（均值）
    m = G.mean(0, keepdims=True)
    mn = m / (np.linalg.norm(m) + 1e-12)
    e_mean = (G @ mn.T) ** 2 / (np.linalg.norm(G, axis=1, keepdims=True) ** 2 + 1e-12)
    # 类子空间
    Q, _ = np.linalg.qr(unitD.T)         # (D, K-1) 正交基
    proj = G @ Q
    e_cls = (proj ** 2).sum(1) / (np.linalg.norm(G, axis=1, keepdims=True) ** 2 + 1e-12)
    return e_mean.ravel(), e_cls.ravel()

em_p, ec_p = energy_split(Gp)
em_r, ec_r = energy_split(Gr)
w('')
w('[5] 能量份额分解（z 向量, 归一化后）')
w('    探针概念: 公共均值方向 %.4f | 类子空间(rank %d) %.4f | 其余 %.4f'
  % (em_p.mean(), K - 1, ec_p.mean(), 1 - em_p.mean() - ec_p.mean()))
w('    随机 token: 公共均值方向 %.4f | 类子空间(rank %d) %.4f | 其余 %.4f'
  % (em_r.mean(), K - 1, ec_r.mean(), 1 - em_r.mean() - ec_r.mean()))
w('    注：若"类子空间"份额 ~9% 且"其余"占主体，则与 2811/2812/2813 一致（类别只是稀疏索引）')

# ---------- 残差（去类子空间）与近正交性 ----------
def resid(G):
    Q, _ = np.linalg.qr(unitD.T)
    return G - (G @ Q) @ Q.T

def cosmat(A, B=None):
    B = A if B is None else B
    An = A / (np.linalg.norm(A, axis=1, keepdims=True) + 1e-12)
    Bn = B / (np.linalg.norm(B, axis=1, keepdims=True) + 1e-12)
    return An @ Bn.T

idx_of = {x: i for i, x in enumerate(probe_ok)}
Cr = cosmat(Gp)
Rr = cosmat(resid(Gp))

w('')
w('[6] 直接回答用户点名的关系对  (cos / 去类残差后 cos)')
w('    关系对              raw_cos   resid_cos')
for a, b in PAIRS:
    if a in idx_of and b in idx_of:
        i, j = idx_of[a], idx_of[b]
        w('    %-6s-%-6s      %+0.4f    %+0.4f' % (a, b, Cr[i, j], Rr[i, j]))

# 随机对基线
RR = cosmat(Gr)
iu = np.triu_indices(RAND_N, 1)
base = RR[iu]
w('')
w('[7] 随机 token 对基线（n=%d 对）: mean %+0.4f  sd %.4f  |cos| mean %.4f  q95 %.4f'
  % (len(base), base.mean(), base.std(), np.abs(base).mean(), np.quantile(np.abs(base), 0.95)))

# 探针内部全部对
iu2 = np.triu_indices(n_p, 1)
pp = Cr[iu2]
w('    探针概念间全部对（n=%d）: mean %+0.4f  |cos| mean %.4f' % (len(pp), pp.mean(), np.abs(pp).mean()))

# 关系对 vs 随机 的对比统计：相关内容对是否显著更高
rel = [Cr[idx_of[a], idx_of[b]] for a, b in PAIRS if a in idx_of and b in idx_of]
w('')
w('[8] 判决性对比：用户认为"相关"的概念对是否比随机对更近？')
w('    相关对 n=%d  mean cos %+0.4f   |cos| mean %.4f' % (len(rel), np.mean(rel), np.mean(np.abs(rel))))
w('    随机对 n=%d  mean cos %+0.4f   |cos| mean %.4f' % (len(base), base.mean(), np.abs(base).mean()))
w('    差值 |cos|: %+0.4f' % (np.mean(np.abs(rel)) - np.abs(base).mean()))
w('    随机对中 |cos| 超过"相关对均值"的比例 = %.3f'
  % float((np.abs(base) > np.mean(np.abs(rel))).mean()))

# 近正交性：残差 pairwise |cos|
w('')
w('[9] 去类后残差的两两 |cos|（2814 核心量）')
w('    探针概念间: |cos| mean %.4f  max %.4f' % (np.abs(Rr[iu2]).mean(), np.abs(Rr[iu2]).max()))
w('    随机 token : |cos| mean %.4f  max %.4f' % (np.abs(base).mean(), np.abs(base).max()))

w('')
w('[10] 多义性结构性事实')
w('      苹果 的 token id = %d（单一 token：水果义与公司义共用同一个嵌入行，几何上不可区分）' % ok['苹果'])
w('      cos(苹果, 水果)=%+0.4f   cos(苹果, 公司)=%+0.4f   cos(苹果, 香蕉)=%+0.4f'
  % (Cr[idx_of['苹果'], idx_of['水果']], Cr[idx_of['苹果'], idx_of['公司']], Cr[idx_of['苹果'], idx_of['香蕉']]))
w('      cos(水果, 食物)=%+0.4f  cos(食物, 物体)=%+0.4f' % (Cr[idx_of['水果'], idx_of['食物']], Cr[idx_of['食物'], idx_of['物体']]))

w('')
w('用时 %.1fs' % (time.time() - t0))

open(OUT, 'w', encoding='utf-8').write('\n'.join(L))
print('WROTE', OUT, len(L), 'lines')
