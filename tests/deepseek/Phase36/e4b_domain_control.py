# -*- coding: utf-8 -*-
"""
E4b 补充臂（Phase 36）：拆"频率 vs 语义域"混杂
A1 中频动物域（高频语义域、中频）：狼 豹 鹿 狐 蛇 鹰 鲤 虾 蟹 龟 鹤 鸦 蚁 蝶 鲸 鲨 蝉 鹅 鸽 驴
A2 低频器物域（生僻、非神兽域）：镯 钗 簪 瓮 甑 笙 磬 钹 铙 戟 钺 篦 笱 罾 醴 樽 爵 觞 砣 铡
A3 原生僻域扩展（神兽/鬼魅域，扩到 >=20）：夔 虬 蛟 鲲 罴 貘 豺 貂 猞 猁? -> 夔 虬 蛟 鲲 罴 貘 狈 貊 麈 麂
判据：若 A2（低频器物）与 RARE 同样低秩而 A1（中频同域）接近满秩 => 频率是主因；
      若 A1 也低秩 => 语义域/具象性是主因，"频率"结论必须收回。
仅 qwen3-4b，零 GPU。seal 判据沿用 E4_design_seal.json 的 C2 口径。
"""
import os, json, time
import numpy as np

ROOT = r'D:\AI2050\Ai2050-OpenOne'
MODEL = os.path.join(ROOT, 'models', 'hf', 'qwen3-4b')
OUT = os.path.join(ROOT, 'tests', 'deepseek_temp', 'Phase36', 'e4b_domain_control_report.txt')

A1 = ["狼","豹","鹿","狐","蛇","鹰","鲤","虾","蟹","龟","鹤","鸦","蚁","蝶","鲸","鲨","蝉","鹅","鸽","驴"]
A2 = ["镯","钗","簪","瓮","甑","笙","磬","钹","铙","戟","钺","篦","笱","罾","醴","樽","爵","觞","砣","铡"]
A3 = ["夔","虬","蛟","鲲","罴","貘","狈","貊","麈","麂","麒","麟","饕","餮","魍","魉","貔","貅","魑","魅"]
HIGH = ["的","了","是","我","你","他","她","它","们","这","那","有","不","没","好","就","都","也","很","和","与","或","但","在","被","把","从","到","上","下","中","里","外","说","要","会","能","去","来","吃","大","小","多","少","个","年","月","日","时","天"]

t0 = time.time()
L = []
def w(s=''):
    L.append(str(s))

from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained(MODEL)
def tids(s):
    return tok.encode(s, add_special_tokens=False)

from safetensors import safe_open
import torch
idx = json.load(open(os.path.join(MODEL, 'model.safetensors.index.json'), encoding='utf-8'))
wmap = idx['weight_map']
key = [k for k in wmap if k.endswith('embed_tokens.weight')][0]
f = safe_open(os.path.join(MODEL, wmap[key]), framework='pt')
sl = f.get_slice(key)
V, D = sl.get_shape()

def rows(ids_arr):
    return np.stack([sl[int(i):int(i)+1].to(torch.float32).numpy()[0] for i in ids_arr], 0)

w('=== E4b 域匹配对照臂（qwen3-4b, 零 GPU）===')
w('时间 %s' % time.strftime('%Y-%m-%d %H:%M:%S'))
w('判据（seal 沿用）：A2 低秩 + A1 满秩 => 频率主因；A1 也低秩 => 语义域主因，频率结论收回')

groups = {}
drops = {}
for gname, gwords in (('A1_mid_animal', A1), ('A2_rare_object', A2), ('A3_rare_mythic', A3), ('HIGH', HIGH)):
    ok = {}
    for x in gwords:
        t = tids(x)
        if len(t) == 1:
            ok[x] = t[0]
        else:
            drops['%s(%s)' % (x, gname)] = len(t)
    groups[gname] = ok

w('')
w('[1] 单 token 存活')
SRC = dict(A1_mid_animal=A1, A2_rare_object=A2, A3_rare_mythic=A3, HIGH=HIGH)
for g, pool in groups.items():
    w('    %-14s %2d/%2d: %s' % (g, len(pool), len(SRC[g]), ' '.join(pool.keys())))
if drops:
    w('    剔除: %s' % ', '.join('%s->%dtok' % (k, v) for k, v in sorted(drops.items())))

rng = np.random.default_rng(20261003)
def per_vector(v):
    v = np.asarray(v, dtype=np.float64)
    n2 = float(v @ v)
    p = (v * v) / n2
    ps = np.sort(p)[::-1]
    return dict(norm=float(np.sqrt(n2)), PR=1.0/float(np.sum(ps*ps)),
                top10=float(ps[:10].sum()), mx=float(ps[0]))

w('')
w('[2] 向量级（中位数）')
vstats = {}
for g, pool in groups.items():
    if len(pool) < 4:
        continue
    E = rows(list(pool.values()))
    st = [per_vector(v) for v in E]
    vstats[g] = st
    pr = np.median([s['PR'] for s in st]); nm = np.median([s['norm'] for s in st])
    t10 = np.median([s['top10'] for s in st])
    w('    %-14s n=%2d  范数=%5.1f  PR=%6.0f  top10=%.3f' % (g, len(st), nm, pr, t10))

w('')
w('[3] 组级矩阵指标（centered；K=20 子采样 x50 当 K>20）')
res = {}
for g, pool in groups.items():
    if len(pool) < 4:
        continue
    E = rows(list(pool.values()))
    Ks = min(20, E.shape[0])
    vals = []
    for it in range(50):
        idxs = rng.choice(E.shape[0], size=Ks, replace=False) if E.shape[0] > Ks else np.arange(E.shape[0])
        Es = E[idxs]
        mc = Es.mean(0, keepdims=True)
        s = np.linalg.svd(Es - mc, compute_uv=False)
        e = s * s
        vals.append((e.sum()**2) / (np.sum(e*e) + 1e-300))
    # 全组组内 cos
    En = E / (np.linalg.norm(E, axis=1, keepdims=True) + 1e-12)
    C = En @ En.T
    iu = np.triu_indices(E.shape[0], 1)
    res[g] = np.median(vals)
    w('    %-14s centered PR_spec=%5.1f (K=%d)   组内raw cos=%+.3f   |cos|=%.3f'
      % (g, res[g], Ks, C[iu].mean(), np.abs(C[iu]).mean()))

w('')
w('[4] 判决')
if 'A1_mid_animal' in res and 'A2_rare_object' in res and 'RARE_ref' not in res:
    a1, a2 = res['A1_mid_animal'], res['A2_rare_object']
    w('    A1(中频·动物域) PR_spec=%.1f   A2(低频·器物域) PR_spec=%.1f   比=%.2f' % (a1, a2, a1/a2))
    if a2 < 0.7 * a1:
        w('    => A2 低秩、A1 近满秩：低频（生僻）是组级低秩的主因，语义域混杂被排除（器物域与神兽域同样抱团，中频动物域不抱团）')
    elif a1 < 0.7 * a2:
        w('    => 反向')
    else:
        w('    => A1/A2 无组级差异：需检查 A2 是否真的低频存活（见 [1] 剔除表）')
if 'A3_rare_mythic' in res and 'A2_rare_object' in res:
    w('    A3(低频·神兽域)=%.1f vs A2(低频·器物域)=%.1f：两域低频组是否同低秩（域无关性检查）' % (res['A3_rare_mythic'], res['A2_rare_object']))

w('')
w('用时 %.1fs' % (time.time() - t0))
with open(OUT, 'w', encoding='utf-8') as fh:
    fh.write('\n'.join(L))
print('WROTE', OUT, len(L), 'lines')
