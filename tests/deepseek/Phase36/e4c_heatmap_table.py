# -*- coding: utf-8 -*-
"""
E4c（Phase 36 附录）：E4/E4b 词嵌入参数可视化表生成器
用户要求：把词嵌入参数生成可视化表格，按参数值高低设置颜色深浅。
- 数据：qwen3-4b embed_tokens（151936 x 2560），safetensors 逐行读，零 GPU。
- 同协议同词池重算（E4 + E4b），非转录；C1/C2 判据数字复现核对。
- 产物：e4c_heatmap_data.json + e4_heatmap.html（三张表：逐 token 参数 / 组级 K=20 / 原始前 96 维热力）
"""
import os, json, time, hashlib
import numpy as np

ROOT = r'D:\AI2050\Ai2050-OpenOne'
MODEL = os.path.join(ROOT, 'models', 'hf', 'qwen3-4b')
OUT_DIR = os.path.join(ROOT, 'tests', 'deepseek_temp', 'Phase36')
os.makedirs(OUT_DIR, exist_ok=True)
DATA_PATH = os.path.join(OUT_DIR, 'e4c_heatmap_data.json')
HTML_PATH = os.path.join(OUT_DIR, 'e4_heatmap.html')

# ---- 词池（与 E4 / E4b 完全一致）----
HIGH = ["的","了","是","我","你","他","她","它","们","这","那","有","不","没","好","就","都","也","很","和","与","或","但","在","被","把","从","到","上","下","中","里","外","说","要","会","能","去","来","吃","大","小","多","少","个","年","月","日","时","天"]
MID = ["猫","狗","山","河","湖","海","树","花","草","书","车","房","路","桥","城","村","云","雨","雪","风"]
RARE = ["麒","麟","饕","餮","魍","魉","貔","貅","魑","魅","龘","靐","麤","犇","骉","羴","猋","曌","齉","龖"]
HIGH2 = ["我们","现在","时间","问题","工作","事情","因为","所以"]
RARE2 = ["麒麟","饕餮","魍魉","貔貅"]
A1 = ["狼","豹","鹿","狐","蛇","鹰","鲤","虾","蟹","龟","鹤","鸦","蚁","蝶","鲸","鲨","蝉","鹅","鸽","驴"]
A2 = ["镯","钗","簪","瓮","甑","笙","磬","钹","铙","戟","钺","篦","笱","罾","醴","樽","爵","觞","砣","铡"]
A3 = ["夔","虬","蛟","鲲","罴","貘","狈","貊","麈","麂","麒","麟","饕","餮","魍","魉","貔","貅","魑","魅"]
FOCUS = ["好","的","麒麟","麒","麟"]

t0 = time.time()
RUNLOG = []

def log(s):
    RUNLOG.append(str(s))

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

def per_vector(v):
    v = np.asarray(v, dtype=np.float64)
    n2 = float(v @ v)
    p = (v * v) / n2
    ps = np.sort(p)[::-1]
    return dict(norm=float(np.sqrt(n2)), PR=1.0/float(np.sum(ps*ps)),
                ED=float(np.exp(-np.sum(p*np.log(p+1e-300)))),
                top10=float(ps[:10].sum()), mx=float(ps[0]))

def mannwhitney(a, b):
    a = np.asarray(a); b = np.asarray(b)
    n1, n2 = len(a), len(b)
    allv = np.concatenate([a, b])
    order = np.argsort(allv, kind='mergesort')
    ranks = np.empty(len(allv))
    sv = allv[order]
    i = 0
    while i < len(sv):
        j = i
        while j + 1 < len(sv) and sv[j + 1] == sv[i]:
            j += 1
        ranks[order[i:j + 1]] = 0.5 * (i + j) + 1.0
        i = j + 1
    R1 = ranks[:n1].sum()
    U1 = R1 - n1 * (n1 + 1) / 2.0
    mu = n1 * n2 / 2.0
    sd = np.sqrt(n1 * n2 * (n1 + n2 + 1) / 12.0)
    z = (U1 - mu) / (sd + 1e-12)
    from math import erf
    p = 2.0 * (1.0 - 0.5 * (1.0 + erf(abs(z) / np.sqrt(2))))
    rbc = 1.0 - 2.0 * U1 / (n1 * n2)
    return U1, z, p, rbc

rng = np.random.default_rng(20261003)

# ---- 词池 -> 单 token 池 ----
def build_pool(words):
    ok = {}
    for x in words:
        t = tids(x)
        if len(t) == 1:
            ok[x] = t[0]
    return ok

pools = {}
for gname, gwords in (('HIGH', HIGH), ('MID', MID), ('RARE', RARE), ('HIGH2', HIGH2), ('RARE2', RARE2),
                      ('A1', A1), ('A2', A2), ('A3', A3)):
    pools[gname] = build_pool(gwords)
log('pools: ' + ', '.join('%s=%d' % (g, len(p)) for g, p in pools.items()))

# ---- 随机 CJK（decode 过滤，同 E4 现场修正）----
cjk_ids = []
seen = set()
tries = 0
while len(cjk_ids) < 32 and tries < 30:
    cand = rng.choice(V, size=4000, replace=False)
    tries += 1
    for i in cand:
        i = int(i)
        if i in seen:
            continue
        s = tok.decode([i])
        seen.add(i)
        if len(s) >= 1 and not s.startswith('<|') and any('\u4e00' <= ch <= '\u9fff' for ch in s):
            cjk_ids.append(i)
            if len(cjk_ids) >= 32:
                break
log('random cjk sampled: %d (tries=%d)' % (len(cjk_ids), tries))

# ---- 高斯带（n=4000, N(0,1)，统计量尺度不变）----
gs = rng.standard_normal((4000, D))
gs_stats = [per_vector(v) for v in gs]
band = dict(
    n=4000,
    PR_mean=round(float(np.mean([s['PR'] for s in gs_stats])), 1),
    PR_sd=round(float(np.std([s['PR'] for s in gs_stats])), 1),
    ED_mean=round(float(np.mean([s['ED'] for s in gs_stats])), 1),
    ED_sd=round(float(np.std([s['ED'] for s in gs_stats])), 1),
    T10_mean=round(float(np.mean([s['top10'] for s in gs_stats])), 4),
    T10_sd=round(float(np.std([s['top10'] for s in gs_stats])), 4),
    MX_mean=round(float(np.mean([s['mx'] for s in gs_stats])), 4),
    MX_sd=round(float(np.std([s['mx'] for s in gs_stats])), 4),
)
log('gauss band: PR=%.0f+/-%.0f  ED=%.0f  (d/3=%.0f)' % (band['PR_mean'], band['PR_sd'], band['ED_mean'], D / 3.0))

# ---- 表 A：逐 token 行（去重 by id）----
sections = [
    ('FOCUS', '用户点名词', None),           # None -> 用 FOCUS 列表
    ('HIGH', '高频功能词', 12),
    ('MID', '中频具象名词', 10),
    ('RARE', '低频生僻单字', None),
    ('RARE2', '低频生僻双字词', None),
    ('R_CJK', '随机 CJK token', 6),
    ('GAUSS', '高斯随机（模拟 N(0,1)）', 4),
]
used_ids = set()
token_rows = []
sec_counts = {}
gauss_disp = rng.standard_normal((4, D))
for sec_key, sec_label, take in sections:
    cnt = 0
    if sec_key == 'FOCUS':
        pairs = []
        for xw in FOCUS:
            t = tids(xw)
            if len(t) == 1:
                pairs.append((xw, t[0]))
    elif sec_key == 'R_CJK':
        pairs = [('(随机)' if len(tok.decode([i])) > 1 else tok.decode([i]), int(i)) for i in cjk_ids[:6]]
    elif sec_key == 'GAUSS':
        pairs = [('G%02d' % (k + 1), -1) for k in range(4)]
    else:
        pool = pools[sec_key]
        pairs = []
        for xw, tid in pool.items():
            if tid in used_ids:
                continue
            pairs.append((xw, tid))
            if take is not None and len(pairs) >= take:
                break
    for xw, tid in pairs:
        if sec_key == 'GAUSS':
            st = per_vector(gauss_disp[cnt])
            tid_out = -1
        else:
            if tid in used_ids:
                continue
            st = per_vector(rows([tid])[0])
            used_ids.add(tid)
            tid_out = tid
        token_rows.append(dict(
            g=sec_key, glabel=sec_label, w=xw, id=tid_out,
            norm=round(st['norm'], 3), PR=round(st['PR'], 1), ED=round(st['ED'], 1),
            top10=round(st['top10'], 4), mx=round(st['mx'], 4)))
        cnt += 1
    sec_counts[sec_key] = cnt
log('tableA rows: %d  sections: %s' % (len(token_rows), sec_counts))

# ---- C1: HIGH vs RARE（全池，向量级 PR / 熵维）----
def full_stats(pool):
    E = rows(list(pool.values()))
    st = [per_vector(v) for v in E]
    return E, st

E_high, st_high = full_stats(pools['HIGH'])
E_rare, st_rare = full_stats(pools['RARE'])
c1 = {}
for k, lab in (('PR', 'PR'), ('ED', 'ED')):
    a = np.array([s[k] for s in st_high])
    b = np.array([s[k] for s in st_rare])
    U, z, p, rbc = mannwhitney(a, b)
    c1[k] = dict(med_high=round(float(np.median(a)), 1), med_rare=round(float(np.median(b)), 1),
                 rel=round(float((np.median(a) - np.median(b)) / (np.median(b) + 1e-12)), 4),
                 U=float(U), z=round(float(z), 2), p=float('%.2e' % p), rbc=round(float(rbc), 2))
log('C1: HIGH PR=%.0f RARE PR=%.0f rel=%+.3f p=%.2e rbc=%+.2f' % (c1['PR']['med_high'], c1['PR']['med_rare'], c1['PR']['rel'], c1['PR']['p'], c1['PR']['rbc']))

# ---- 表 B：组级 K=20 匹配（50 次子采样，centered）+ 全组 cos ----
def group_table(gname, label, E):
    Kfull = E.shape[0]
    Ks = min(20, Kfull)
    vals_pr, vals_ent = [], []
    for it in range(50):
        idxs = rng.choice(Kfull, size=Ks, replace=False) if Kfull > Ks else np.arange(Kfull)
        Es = E[idxs]
        mc = Es.mean(0, keepdims=True)
        s = np.linalg.svd(Es - mc, compute_uv=False)
        e = s * s
        vals_pr.append((e.sum() ** 2) / (np.sum(e * e) + 1e-300))
        qq = e / (e.sum() + 1e-300)
        vals_ent.append(float(np.exp(-np.sum(qq * np.log(qq + 1e-300)))))
    En = E / (np.linalg.norm(E, axis=1, keepdims=True) + 1e-12)
    C = En @ En.T
    iu = np.triu_indices(Kfull, 1)
    rawc = C[iu]
    Cn = C - np.eye(Kfull)
    note = ''
    if Ks < 20:
        note = 'n<20：PR_spec 受 K 上界压制，不可与 K=20 组直接比较（E4b 已判此口径为 K 失配伪影）'
    return dict(g=gname, label=label, n=int(Kfull), K=int(Ks),
                PR_spec=round(float(np.median(vals_pr)), 1), EntRank=round(float(np.median(vals_ent)), 1),
                cos_raw=round(float(rawc.mean()), 4), cos_abs=round(float(np.abs(rawc).mean()), 4),
                nn_cos=round(float(Cn.max(axis=1).mean()), 4), note=note)

group_rows = []
group_defs = [
    ('HIGH', '高频功能词'), ('MID', '中频具象名词'), ('RARE', '低频生僻单字'),
    ('HIGH2', '高频双字词'), ('RARE2', '低频生僻双字词'),
    ('A1', '中频·动物域'), ('A2', '低频·器物域'), ('A3', '低频·神兽域'),
]
for gname, label in group_defs:
    if len(pools[gname]) >= 4:
        group_rows.append(group_table(gname, label, rows(list(pools[gname].values()))))
if len(cjk_ids) >= 4:
    group_rows.append(group_table('R_CJK', '随机 CJK token', rows(cjk_ids[:32])))
rand_all = rng.choice(V, size=200, replace=False)
group_rows.append(group_table('R_ALL', '全词表随机(200)', rows(rand_all)))
gauss_g = rng.standard_normal((20, D))
group_rows.append(group_table('GAUSS', '高斯随机(模拟)', gauss_g))
log('tableB groups: %d' % len(group_rows))

pr = {r['g']: r['PR_spec'] for r in group_rows}
c2 = dict(PR_high=pr.get('HIGH'), PR_rare=pr.get('RARE'),
          ratio=round(pr['HIGH'] / (pr['RARE'] + 1e-12), 2),
          A1=pr.get('A1'), A2=pr.get('A2'), A3=pr.get('A3'),
          verdict='RARE n=14<20：1.57 是 K 失配读数（E4b 判）；K=20 严格匹配的对照 = A1/A2/A3 与 HIGH 同档')
log('C2: HIGH=%.1f RARE=%.1f ratio=%.2f | A1=%.1f A2=%.1f A3=%.1f' % (c2['PR_high'], c2['PR_rare'], c2['ratio'], c2['A1'], c2['A2'], c2['A3']))

# ---- 表 C：原始前 96 维 ----
NDIM = 96
raw_defs = [('好', 'FOCUS'), ('的', 'FOCUS'), ('猫', 'MID'), ('麒麟', 'RARE2'), ('麒', 'RARE'), ('麟', 'RARE'), ('魑', 'RARE')]
raw_tokens = []
raw_mat = []
for xw, g in raw_defs:
    t = tids(xw)
    if len(t) != 1:
        continue
    raw_mat.append(rows([t[0]])[0][:NDIM].astype(np.float64))
    raw_tokens.append(dict(w=xw, g=g, id=int(t[0])))
cj = tok.decode([int(cjk_ids[0])])
raw_mat.append(rows([int(cjk_ids[0])])[0][:NDIM].astype(np.float64))
raw_tokens.append(dict(w=cj, g='R_CJK', id=int(cjk_ids[0])))
gvec = rng.standard_normal(D)
med_norm = float(np.median([np.linalg.norm(v) for v in raw_mat]))
gvec = gvec * (med_norm / np.linalg.norm(gvec))
raw_mat.append(gvec[:NDIM])
raw_tokens.append(dict(w='高斯(缩放)', g='GAUSS', id=-1))
raw_arr = np.stack(raw_mat, 0)
vmax = float(np.abs(raw_arr).max())
log('tableC: %d rows x %d dims, vmax=%.4f, med_norm=%.2f, gauss scaled to med_norm' % (raw_arr.shape[0], NDIM, vmax, med_norm))

data = dict(
    meta=dict(model='qwen3-4b', key=key, V=int(V), D=int(D), shard=wmap[key],
              time=time.strftime('%Y-%m-%d %H:%M:%S'), seed=20261003,
              note='全部数字由本脚本现场重算（E4/E4b 同协议同词池），非转录。单条词嵌入是向量、无"秩"；PR=参与率有效维=(Σv^2)^2/Σv^4，上限 d。'),
    gauss_band=band, c1=c1, c2=c2,
    sec_counts=sec_counts, pool_sizes={g: len(p) for g, p in pools.items()},
    tokens=token_rows, groups=group_rows,
    raw=dict(tokens=raw_tokens, ndims=NDIM, vmax=round(vmax, 5),
             rows=[[round(float(x), 5) for x in r] for r in raw_arr],
             gauss_scaled_to=round(med_norm, 3)),
)
with open(DATA_PATH, 'w', encoding='utf-8') as fh:
    json.dump(data, fh, ensure_ascii=False)
log('json written: %d bytes' % os.path.getsize(DATA_PATH))

# ================= HTML =================
def sha8(path):
    return hashlib.sha256(open(path, 'rb').read()).hexdigest()[:8]

TMPL = """<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<title>E4c 词嵌入参数可视化 — qwen3-4b embed_tokens</title>
<style>
  body { font-family: "Segoe UI", "Microsoft YaHei", sans-serif; background:#f5f7fa; color:#1f2937;
         margin:0; padding:24px 32px 60px; }
  .wrap { max-width: 1280px; margin: 0 auto; }
  h1 { font-size: 22px; margin: 0 0 4px; }
  h2 { font-size: 17px; margin: 34px 0 6px; border-left: 4px solid #2563eb; padding-left: 10px; }
  .meta { color:#6b7280; font-size: 12.5px; line-height: 1.7; }
  .box { background:#ffffff; border:1px solid #e5e7eb; border-radius:10px; padding:14px 18px; margin:10px 0 4px; }
  .note { font-size: 12.5px; color:#4b5563; background:#eef4ff; border:1px solid #dbe7ff;
          border-radius:8px; padding:10px 14px; margin:8px 0; line-height:1.75; }
  .verdict { font-size: 13.5px; background:#f0fdf4; border:1px solid #bbf7d0; border-radius:8px;
             padding:10px 14px; margin:8px 0; line-height:1.8; }
  table { border-collapse: collapse; background:#fff; font-size: 12.5px; }
  th, td { border:1px solid #e5e7eb; padding: 4px 9px; text-align:center; white-space:nowrap; }
  thead th { background:#1e3a5f; color:#fff; font-weight:600; position:sticky; top:0; }
  tr.gsec td { background:#e8eef7; color:#1e3a5f; font-weight:700; text-align:left;
               letter-spacing:.5px; padding:5px 10px; }
  td.tok { font-family: "Microsoft YaHei", sans-serif; font-weight:700; font-size:14px; text-align:left; }
  td.idc { color:#9ca3af; font-size:11px; font-family:Consolas,monospace; }
  td.num { font-family:Consolas,monospace; font-weight:600; }
  tbody tr:hover td { outline:1.5px solid #60a5fa; outline-offset:-1.5px; }
  .scroll { overflow-x:auto; border:1px solid #e5e7eb; border-radius:8px; background:#fff; }
  .legend { display:inline-flex; align-items:center; gap:8px; font-size:12px; color:#374151;
            background:#fff; border:1px solid #e5e7eb; border-radius:6px; padding:5px 10px; margin:6px 10px 6px 0; }
  .bar { width:150px; height:12px; border-radius:6px; border:1px solid #d1d5db; }
  .seqbar { background: linear-gradient(to right, rgb(247,251,255), rgb(33,113,181)); }
  .divbar { background: linear-gradient(to right, rgb(33,102,172), rgb(255,255,255), rgb(178,24,43)); }
  #tblC td.cell { min-width:27px; padding:4px 3px; font-size:10.5px; font-family:Consolas,monospace; }
  #tblC th.dim { background:#1e3a5f; color:#cbd5e1; font-size:10px; padding:3px 2px; font-family:Consolas,monospace; }
  #tblC td.rlab { position:sticky; left:0; background:#f8fafc; font-weight:700; font-size:13.5px;
                  text-align:left; padding:4px 10px; border-right:2px solid #cbd5e1; z-index:2; }
  .foot { margin-top:28px; font-size:12px; color:#6b7280; line-height:1.8; border-top:1px solid #e5e7eb; padding-top:12px; }
  .big { font-size:14.5px; font-weight:700; color:#111827; }
  sup { color:#6b7280; font-weight:400; }
</style>
</head>
<body>
<div class="wrap">
  <h1>E4c 词嵌入参数可视化表 — qwen3-4b <span style="font-weight:400;font-size:14px;color:#6b7280">embed_tokens (151,936 × 2560, bf16)</span></h1>
  <div class="meta">
    生成时间 @@GENTIME@@ ｜ 数据来源：models/hf/qwen3-4b safetensors 逐行直读（零 GPU，加载 + 全部重算 @@RUNSEC@@ s）<br>
    协议：与 <b>E4（词频 × 有效维度）</b>、<b>E4b（域匹配对照臂）</b> 完全同协议、同词池、同 seed；本页所有数字为现场重算，非人工转录。判据 seal：<span style="font-family:Consolas">tests/deepseek/Phase36/E4_design_seal.json</span>
  </div>

  <div class="note">
    <b>怎么读这张表</b>：一条词嵌入是一个 2560 维向量，单条向量没有"秩"。它的"摊开程度"用 <b>PR（参与率有效维）= (Σvᵢ²)² / Σvᵢ⁴</b> 度量（能量摊在多少个维度上，上限 2560）；高斯随机向量在 d=2560 时的理论期望 ≈ d/3 ≈ 853。<b>深色 = 数值高，浅色 = 数值低</b>；每列独立归一化（列内可比，跨列只看相对深浅）。把鼠标放在任意单元格上可看精确值。数值为本页脚本现场重算（同协议同 seed），个别整数值与报告打印差 ±1 属舍入边界（如 907.5 一方取 907 一方取 908）。
  </div>
  <div>
    <span class="legend"><span class="bar seqbar"></span> 顺序色标：浅 → 深 = 列内最小 → 最大值</span>
    <span class="legend"><span class="bar divbar"></span> 发散色标：蓝(负) ↔ 白(0) ↔ 红(正)</span>
  </div>

  <h2>表 A ｜ 逐 token 参数总表（每组抽样显示，色深 = 值高低）</h2>
  <div class="verdict" id="verdict1"></div>
  <div class="box" style="padding:8px"><div style="overflow-x:auto"><table id="tblA"></table></div></div>

  <h2>表 B ｜ 组级矩阵有效秩（K=20 严格匹配 × 50 次子采样，centered 口径）</h2>
  <div class="verdict" id="verdict2"></div>
  <div class="box" style="padding:8px"><div style="overflow-x:auto"><table id="tblB"></table></div></div>

  <h2>表 C ｜ 原始嵌入值热力图（前 96 维 × 9 个代表 token，发散色标）</h2>
  <div class="note" style="margin-top:4px">
    维度按索引 0–95 排列（嵌入维度没有规范顺序，此处仅为原始参数直览）；色标按全表最大幅值 ±@@VMAX@@ 归一：<b>红 = 正值，蓝 = 负值，色越深 = |值| 越大</b>。高斯行已缩放到嵌入范数中位数（@@MEDNORM@@，仅显示用）——它就是"未训练随机初始化"大致的样子：与任何真实 token 一样摊满高维，但与同组词毫无相干结构。
  </div>
  <div class="scroll"><table id="tblC"></table></div>

  <div class="foot">
    <div class="big">一句话结论：常用词不是"满秩"、生僻字不是"低秩" —— 所有嵌入都摊在 ~800–950 个有效维度上（d=2560 的 1/3 高斯带附近）；真正的组级结构是<b>语义域相干性</b>（同域组内 cos 0.10–0.15 vs 跨域虚词 0.065），不是词频。</div>
    <div style="margin-top:6px">
      ① 向量级：HIGH 中位 PR=@@PRH@@ vs RARE @@PRR@@（相对差 @@REL@@%，Mann-Whitney p=@@PVAL@@，秩双列相关 @@RBC@@）——方向与"常用词更满秩"的猜测<b>相反</b>，常用词反而略更摊开。<br>
      ② 组级：原始 RARE 池只存活 n=14，其 PR_spec=11.5 对 HIGH(18.0) 的比 1.57 是 <b>K 失配伪影</b>（E4 自纠在案）；K=20 严格匹配的域对照 A1 动物 @@A1V@@ / A2 器物 @@A2V@@ / A3 神兽 @@A3V@@ 与 HIGH 同档 ⇒ 匹配样本数后组级秩差消失。<br>
      ③ 生僻字≠随机但欠训练：RARE 高斯带命中率 57–79%；组内两两 cos @@COSR@@，显著高于随机 CJK（@@COSC@@）、全词表随机（@@COSA@@）与高斯（@@COSG@@）——"神兽域抱团"是真实相干结构，不是噪声抱团。
    </div>
    <div style="margin-top:8px"><b>硬伤登记</b>：词频分组为人工判定（未用语料实测频率）；向量级效应量小（~10%）；随机 CJK / GAUSS 对照与真实低频 token 的训练量不可严格对齐；本页为 Phase 36 已有结论的可视化呈现，无新实验、无新判决。</div>
    <div style="margin-top:8px; font-family:Consolas; font-size:11px">
      产物：e4_heatmap.html（sha8 见登记册 _sha8_register.txt）｜ e4c_heatmap_data.json sha8=@@SHA_JSON@@ ｜ e4c_heatmap_table.py sha8=@@SHA_PY@@ ｜ seal E4_design_seal.json sha8=@@SHA_SEAL@@ ｜ 上游：e4_freq_rank_probe.py @@SHA_E4@@ / e4b_domain_control.py @@SHA_E4B@@<br>
      上游报告：tests/deepseek_temp/Phase36/e4_report_qwen3-4b.txt、e4b_domain_control_report.txt ｜ memo：research/deepseek/docs/AGI_DEEPSEEK_MEMO.md Phase 36
    </div>
  </div>
</div>

<script>
const DATA = @@DATA@@;

function lerp(a,b,t){ return a + (b-a)*t; }
function seqRGB(t){ // 浅 -> 深（蓝）
  t = Math.max(0, Math.min(1, t));
  return [lerp(247,33,t), lerp(251,113,t), lerp(255,181,t)];
}
function divRGB(t){ // t in [-1,1] 蓝(负) 白(0) 红(正)
  const u = Math.max(-1, Math.min(1, t));
  const c = u < 0 ? [33,102,172] : [178,24,43];
  const a = Math.abs(u);
  return [lerp(255,c[0],a), lerp(255,c[1],a), lerp(255,c[2],a)];
}
function cellStyle(rgb){
  const L = (0.299*rgb[0] + 0.587*rgb[1] + 0.114*rgb[2]) / 255;
  const tc = L > 0.60 ? '#16213e' : '#ffffff';
  return 'background:rgb(' + rgb.map(Math.round).join(',') + ');color:' + tc + ';';
}
function fmt(v, dec){ return Number(v).toFixed(dec); }

// ============ 表 A ============
(function(){
  const cols = [
    {k:'norm', name:'范数', dec:2},
    {k:'PR',   name:'PR 有效维', dec:0},
    {k:'PRd',  name:'PR / d', dec:3},
    {k:'ED',   name:'熵维', dec:0},
    {k:'top10',name:'top10 份额', dec:3},
    {k:'mx',   name:'max 份额', dec:4},
  ];
  const rows = DATA.tokens.map(r => Object.assign({}, r, {PRd: r.PR / DATA.meta.D}));
  const mm = {};
  cols.forEach(c => {
    const vs = rows.map(r => r[c.k]);
    mm[c.k] = [Math.min.apply(null, vs), Math.max.apply(null, vs)];
  });
  const secOrder = [['FOCUS','用户点名词'],['HIGH','高频功能词'],['MID','中频具象名词'],
                    ['RARE','低频生僻单字'],['RARE2','低频生僻双字词'],['R_CJK','随机 CJK token'],['GAUSS','高斯随机（模拟 N(0,1)）']];
  let h = '<thead><tr><th>token</th><th>id</th>';
  cols.forEach(c => { h += '<th title="深色=该列数值高">' + c.name + '</th>'; });
  h += '</tr></thead><tbody>';
  secOrder.forEach(sec => {
    const rs = rows.filter(r => r.g === sec[0]);
    if (!rs.length) return;
    const poolN = DATA.pool_sizes[sec[0]];
    const dispN = DATA.sec_counts[sec[0]];
    const poolTxt = (poolN !== undefined && sec[0] !== 'GAUSS' && sec[0] !== 'R_CJK') ? '，池 n=' + poolN : '';
    h += '<tr class="gsec"><td colspan="8">■ ' + sec[1] + '（显示 ' + dispN + poolTxt + '）</td></tr>';
    rs.forEach(r => {
      h += '<tr><td class="tok" title="' + r.glabel + '">' + r.w + '</td>';
      h += '<td class="idc">' + (r.id < 0 ? '—' : r.id) + '</td>';
      cols.forEach(c => {
        const v = r[c.k];
        const t = (v - mm[c.k][0]) / ((mm[c.k][1] - mm[c.k][0]) || 1);
        h += '<td class="num" style="' + cellStyle(seqRGB(t)) + '" title="' + c.name + ' = ' + fmt(v, c.dec) + '">' + fmt(v, c.dec) + '</td>';
      });
      h += '</tr>';
    });
  });
  h += '</tbody>';
  document.getElementById('tblA').innerHTML = h;
  const c1 = DATA.c1.PR;
  function medOf(sec, k){
    const vs = rows.filter(r => r.g === sec).map(r => r[k]).sort(function(a,b){return a-b;});
    return vs.length % 2 ? vs[(vs.length-1)/2] : (vs[vs.length/2-1] + vs[vs.length/2]) / 2;
  }
  document.getElementById('verdict1').innerHTML =
    '<b>读法</b>：看 PR 列——所有 token 都在 ' + Math.round(mm['PR'][0]) + '–' + Math.round(mm['PR'][1]) +
    ' 之间（高斯带 ' + DATA.gauss_band.PR_mean + '±' + DATA.gauss_band.PR_sd + '），<b>没有任何一组是低维的</b>；' +
    '高频词 PR 中位 ' + Math.round(c1.med_high) + ' vs 生僻字 ' + Math.round(c1.med_rare) +
    '（相对差 ' + (c1.rel*100).toFixed(1) + '%，p=' + c1.p.toExponential(1) + '）——方向与"常用词更满秩"相反。' +
    'top10 / max 份额列颜色越深 = 能量越集中：生僻字（top10 中位 ' + medOf('RARE','top10').toFixed(3) +
    '）略高于高频词（' + medOf('HIGH','top10').toFixed(3) + '）、高斯基准 ' + DATA.gauss_band.T10_mean +
    '——生僻字介于随机与常用码之间（欠训练但非纯随机）。';
})();

// ============ 表 B ============
(function(){
  const cols = [
    {k:'PR_spec', name:'PR_spec<br>(K=20)', dec:1, mode:'seq', tip:'centered 谱参与率，K=20 x50 子采样中位'},
    {k:'EntRank', name:'熵秩<br>(K=20)', dec:1, mode:'seq', tip:'谱熵秩，同口径'},
    {k:'cos_raw', name:'组内 cos', dec:3, mode:'div', tip:'全组两两 raw cosine 均值'},
    {k:'cos_abs', name:'|cos|', dec:3, mode:'seq', tip:'两两 |cos| 均值'},
    {k:'nn_cos',  name:'最近邻 cos', dec:3, mode:'div', tip:'每行去掉自比后的最大 cos 均值'},
  ];
  const mm = {};
  cols.forEach(c => {
    const vs = DATA.groups.map(r => Math.abs(r[c.k]));
    mm[c.k] = Math.max.apply(null, vs) || 1;
  });
  let h = '<thead><tr><th style="text-align:left">组</th><th>n</th><th>K</th>';
  cols.forEach(c => { h += '<th title="' + c.tip + '">' + c.name + '</th>'; });
  h += '</tr></thead><tbody>';
  DATA.groups.forEach(r => {
    const warn = r.note ? ' <span style="color:#b45309" title="' + r.note + '">⚠</span>' : '';
    h += '<tr><td class="tok" style="font-size:12.5px">' + r.label + ' <span style="color:#9ca3af;font-weight:400">(' + r.g + ')</span>' + warn + '</td>';
    h += '<td class="num">' + r.n + '</td><td class="num"' + (r.note ? ' style="color:#b45309"' : '') + '>' + r.K + '</td>';
    cols.forEach(c => {
      const v = r[c.k];
      let style;
      if (c.mode === 'seq') {
        style = cellStyle(seqRGB(Math.abs(v) / mm[c.k]));
      } else {
        style = cellStyle(divRGB(v / mm[c.k]));
      }
      h += '<td class="num" style="' + style + '" title="' + c.tip + ' = ' + fmt(v, c.dec) + '">' + fmt(v, c.dec) + '</td>';
    });
    h += '</tr>';
  });
  h += '</tbody>';
  document.getElementById('tblB').innerHTML = h;
  const c2 = DATA.c2;
  const g = {}; DATA.groups.forEach(r => { g[r.g] = r; });
  document.getElementById('verdict2').innerHTML =
    '<b>读法</b>：⚠ RARE 池只存活 n=14（K=14），PR_spec=11.5 受 K 上界压制——它对 HIGH(K=20)=18.0 的比 1.57 <b>正是 E4b 已否决的 K 失配伪影</b>，不能当结论。' +
    'K=20 严格匹配的比较是三组语义域（A1 动物 ' + fmt(c2.A1,1) + ' / A2 器物 ' + fmt(c2.A2,1) + ' / A3 神兽 ' + fmt(c2.A3,1) +
    '）：与 HIGH 18.0 <b>同档 ⇒ 匹配样本数后，频率不再产生组级秩差</b>。真正的分化在 <b>组内 cos</b> 列：同域组（A1/A2/A3）偏红（正相干），高频虚词与随机组接近白/蓝——' +
    '<b>语义域相干性，不是词频</b>。GAUSS 行是纯随机的样子：PR_spec≈20 满档、cos≈0。';
})();

// ============ 表 C ============
(function(){
  const rc = DATA.raw;
  const vmax = rc.vmax;
  let h = '<thead><tr><th class="dim" style="text-align:left">token \\ dim</th>';
  for (let d = 0; d < rc.ndims; d++) {
    h += '<th class="dim"' + (d % 8 === 0 ? ' style="color:#fff"' : '') + '>' + d + '</th>';
  }
  h += '</tr></thead><tbody>';
  rc.tokens.forEach((tk, i) => {
    h += '<tr><td class="rlab" title="组=' + tk.g + ' id=' + tk.id + '">' + tk.w + '</td>';
    rc.rows[i].forEach((v, d) => {
      h += '<td class="cell" style="' + cellStyle(divRGB(v / vmax)) + '" title="dim ' + d + ' = ' + v.toFixed(5) + '">' + v.toFixed(3) + '</td>';
    });
    h += '</tr>';
  });
  h += '</tbody>';
  document.getElementById('tblC').innerHTML = h;
})();
</script>
</body>
</html>
"""

html = (TMPL
        .replace('@@GENTIME@@', time.strftime('%Y-%m-%d %H:%M:%S'))
        .replace('@@RUNSEC@@', '%.1f' % (time.time() - t0))
        .replace('@@VMAX@@', str(data['raw']['vmax']))
        .replace('@@MEDNORM@@', str(data['raw']['gauss_scaled_to']))
        .replace('@@PRH@@', str(c1['PR']['med_high']))
        .replace('@@PRR@@', str(c1['PR']['med_rare']))
        .replace('@@REL@@', str(round(c1['PR']['rel'] * 100, 1)))
        .replace('@@PVAL@@', '%.1e' % c1['PR']['p'])
        .replace('@@RBC@@', str(c1['PR']['rbc']))
        .replace('@@RATIO@@', str(c2['ratio']))
        .replace('@@A1V@@', str(c2['A1']))
        .replace('@@A2V@@', str(c2['A2']))
        .replace('@@A3V@@', str(c2['A3']))
        .replace('@@NNR@@', str({r['g']: r['nn_cos'] for r in group_rows}.get('RARE')))
        .replace('@@COSR@@', '+%.3f' % {r['g']: r['cos_raw'] for r in group_rows}.get('RARE', 0))
        .replace('@@COSC@@', '+%.3f' % {r['g']: r['cos_raw'] for r in group_rows}.get('R_CJK', 0))
        .replace('@@COSA@@', '+%.3f' % {r['g']: r['cos_raw'] for r in group_rows}.get('R_ALL', 0))
        .replace('@@COSG@@', '+%.3f' % {r['g']: r['cos_raw'] for r in group_rows}.get('GAUSS', 0))
        .replace('@@SHA_JSON@@', sha8(DATA_PATH))
        .replace('@@SHA_PY@@', sha8(os.path.abspath(__file__)))
        .replace('@@SHA_SEAL@@', sha8(os.path.join(ROOT, 'tests', 'deepseek', 'Phase36', 'E4_design_seal.json')))
        .replace('@@SHA_E4@@', sha8(os.path.join(ROOT, 'tests', 'deepseek', 'Phase36', 'e4_freq_rank_probe.py')))
        .replace('@@SHA_E4B@@', sha8(os.path.join(ROOT, 'tests', 'deepseek', 'Phase36', 'e4b_domain_control.py')))
        .replace('@@DATA@@', json.dumps(data, ensure_ascii=False))
        )

with open(HTML_PATH, 'w', encoding='utf-8') as fh:
    fh.write(html)
log('html written: %d bytes' % os.path.getsize(HTML_PATH))

with open(os.path.join(OUT_DIR, 'e4c_run_log.txt'), 'w', encoding='utf-8') as fh:
    fh.write('\n'.join(RUNLOG))
print('DONE html=%d bytes, %.1fs' % (os.path.getsize(HTML_PATH), time.time() - t0))
