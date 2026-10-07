# -*- coding: utf-8 -*-
"""
E4 探索性探针（Phase 36）：词频 x 嵌入有效维度
用户问题：常用词（好/的）的词嵌入是否"满秩"，生僻字（麒麟）是否"低秩"？
操作化：A 向量级有效维度（PR/熵维/top10/max）；B 组级矩阵有效秩（SVD 谱，raw+centered）。
零 GPU：safetensors 逐行读 embed_tokens。协议对齐 E1（2026-09-30）。
seal: tests/deepseek/Phase36/E4_design_seal.json（观测前冻结）
"""
import os, json, time
import numpy as np

ROOT = r'D:\AI2050\Ai2050-OpenOne'
MODELS = [
    ('qwen3-4b', os.path.join(ROOT, 'models', 'hf', 'qwen3-4b')),
    ('qwen2.5-3b-instruct', os.path.join(ROOT, 'models', 'hf', 'qwen2.5-3b-instruct')),
    ('glm4-9b-chat-hf', os.path.join(ROOT, 'models', 'hf', 'glm4-9b-chat-hf')),
]
OUT_DIR = os.path.join(ROOT, 'tests', 'deepseek_temp', 'Phase36')
os.makedirs(OUT_DIR, exist_ok=True)

HIGH = ["的","了","是","我","你","他","她","它","们","这","那","有","不","没","好","就","都","也","很","和","与","或","但","在","被","把","从","到","上","下","中","里","外","说","要","会","能","去","来","吃","大","小","多","少","个","年","月","日","时","天"]
MID = ["猫","狗","山","河","湖","海","树","花","草","书","车","房","路","桥","城","村","云","雨","雪","风"]
RARE = ["麒","麟","饕","餮","魍","魉","貔","貅","魑","魅","龘","靐","麤","犇","骉","羴","猋","曌","齉","龖"]
HIGH2 = ["我们","现在","时间","问题","工作","事情","因为","所以"]
RARE2 = ["麒麟","饕餮","魍魉","貔貅"]
FOCUS = ["好","的","麒麟","麒","麟"]   # 用户点名

t_start = time.time()

def per_vector(v):
    v = np.asarray(v, dtype=np.float64)
    n2 = float(v @ v)
    if n2 <= 0:
        return None
    p = (v * v) / n2
    ps = np.sort(p)[::-1]
    pr = 1.0 / float(np.sum(ps * ps))
    ent = float(-np.sum(p * np.log(p + 1e-300)))
    return dict(norm=float(np.sqrt(n2)), PR=pr, ED=float(np.exp(ent)),
                top10=float(ps[:10].sum()), top64=float(ps[:64].sum()), mx=float(ps[0]))

def group_metrics(E):
    """E: (K,d) float64. returns dict incl raw/centered spectra + cos stats."""
    K = E.shape[0]
    En = E / (np.linalg.norm(E, axis=1, keepdims=True) + 1e-12)
    C = En @ En.T
    iu = np.triu_indices(K, 1)
    raw_cos = C[iu]
    mc = E.mean(0, keepdims=True)
    Ec = E - mc
    Ecn = Ec / (np.linalg.norm(Ec, axis=1, keepdims=True) + 1e-12)
    Cc = Ecn @ Ecn.T
    cen_cos = Cc[iu]
    out = {}
    for tag, M in (('raw', E), ('centered', Ec)):
        s = np.linalg.svd(M, compute_uv=False)
        e = s * s
        tot = e.sum() + 1e-300
        pr = (e.sum() ** 2) / (np.sum(e * e) + 1e-300)
        q = e / tot
        ent = float(-np.sum(q * np.log(q + 1e-300)))
        out[tag] = dict(PR_spec=float(pr), EntRank=float(np.exp(ent)),
                        num_rank_1e2=int((s > 1e-2 * s[0]).sum()),
                        num_rank_1e1=int((s > 1e-1 * s[0]).sum()))
    out['pair_raw'] = dict(mean=float(raw_cos.mean()), absmean=float(np.abs(raw_cos).mean()))
    out['pair_centered'] = dict(mean=float(cen_cos.mean()), absmean=float(np.abs(cen_cos).mean()))
    Cn = C - np.eye(K)
    out['nn_cos'] = float(Cn.max(axis=1).mean())
    return out

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

def is_cjk(s):
    return any('\u4e00' <= ch <= '\u9fff' for ch in s)

def is_format(s):
    return all((not ch.isalnum()) and (ch not in '\u4e00\u4e01') for ch in s) or s.startswith('<|')

for model_name, model_dir in MODELS:
    rep_path = os.path.join(OUT_DIR, 'e4_report_%s.txt' % model_name)
    L = []
    def w(s=''):
        L.append(str(s))
    try:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(model_dir)
    except Exception as e:
        with open(rep_path, 'w', encoding='utf-8') as f:
            f.write('MODEL SKIP: %s (%s)\n' % (model_name, e))
        continue

    def tids(s):
        return tok.encode(s, add_special_tokens=False)

    w('=== E4 词频 x 嵌入有效维度探针：%s ===' % model_name)
    w('时间 %s' % time.strftime('%Y-%m-%d %H:%M:%S'))
    w('seal: tests/deepseek/Phase36/E4_design_seal.json（观测前冻结）')

    from safetensors import safe_open
    import torch
    idx_path = os.path.join(model_dir, 'model.safetensors.index.json')
    shards = {}
    if os.path.exists(idx_path):
        wmap = json.load(open(idx_path, encoding='utf-8'))['weight_map']
        key = [k for k in wmap if k.endswith('embed_tokens.weight')][0]
        shards[key] = wmap[key]
    else:
        for f2 in sorted(os.listdir(model_dir)):
            if f2.endswith('.safetensors'):
                f3 = safe_open(os.path.join(model_dir, f2), framework='pt')
                ks = list(f3.keys())
                for k in ks:
                    if k.endswith('embed_tokens.weight'):
                        shards[k] = f2
        key = list(shards.keys())[0]
    shard = shards[key]
    f = safe_open(os.path.join(model_dir, shard), framework='pt')
    sl = f.get_slice(key)
    V, D = sl.get_shape()
    w('模型 %s  embed: %s  shape=(%d,%d)  shard=%s' % (model_name, key, V, D, shard))

    def rows(ids_arr):
        return np.stack([sl[int(i):int(i) + 1].to(torch.float32).numpy()[0] for i in ids_arr], 0)

    # --- token 过滤 ---
    groups_single = {}
    drops = {}
    for gname, gwords in (('HIGH', HIGH), ('MID', MID), ('RARE', RARE)):
        ok = {}
        for x in gwords:
            t = tids(x)
            if len(t) == 1:
                ok[x] = t[0]
            else:
                drops['%s(%s)' % (x, gname)] = len(t)
        groups_single[gname] = ok
    groups_two = {}
    for gname, gwords in (('HIGH2', HIGH2), ('RARE2', RARE2)):
        ok = {}
        for x in gwords:
            t = tids(x)
            if len(t) == 1:
                ok[x] = t[0]
            else:
                drops['%s(%s)' % (x, gname)] = len(t)
        groups_two[gname] = ok

    w('')
    w('[1] 单 token 存活')
    for g in ('HIGH', 'MID', 'RARE', 'HIGH2', 'RARE2'):
        src = dict(HIGH=HIGH, MID=MID, RARE=RARE, HIGH2=HIGH2, RARE2=RARE2)[g]
        pool = groups_single[g] if g in ('HIGH', 'MID', 'RARE') else groups_two[g]
        w('    %-5s %d/%d 存活: %s' % (g, len(pool), len(src), ' '.join(pool.keys())))
    if drops:
        w('    剔除: %s' % ', '.join('%s->%dtok' % (k, v) for k, v in sorted(drops.items())))

    rng = np.random.default_rng(20261003)
    # 按 decode 结果筛 CJK（Qwen 词表键为 byte-unicode 串，直接按键过滤失效——E4 现场修正）
    cjk_ids = []
    seen = set()
    tries = 0
    while len(cjk_ids) < 200 and tries < 30:
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
                if len(cjk_ids) >= 200:
                    break
    cjk_ids = np.array(sorted(cjk_ids), dtype=np.int64)
    rand_cjk_ids = cjk_ids[:200] if len(cjk_ids) >= 4 else np.array([], dtype=np.int64)
    rand_all_ids = rng.choice(V, size=600, replace=False)

    w('    RANDOM_CJK 池: %d 个 CJK token（decode 过滤，试采 %d 轮）' % (len(cjk_ids), tries))

    # --- 读行 ---
    data = {}
    for g, pool in list(groups_single.items()) + list(groups_two.items()):
        if len(pool) >= 4:
            data[g] = rows(list(pool.values()))
    if len(rand_cjk_ids) >= 4:
        data['RANDOM_CJK'] = rows(rand_cjk_ids)
    data['RANDOM_ALL'] = rows(rand_all_ids)

    # --- 高斯模拟零带 ---
    gs = rng.standard_normal((4000, D))
    gs_stats = [per_vector(v) for v in gs]
    gPR = np.array([s['PR'] for s in gs_stats]); gED = np.array([s['ED'] for s in gs_stats])
    gT10 = np.array([s['top10'] for s in gs_stats]); gMX = np.array([s['mx'] for s in gs_stats])
    w('')
    w('[2] 高斯随机向量模拟带（n=4000, d=%d）' % D)
    w('    PR: %.0f +/- %.0f   熵维: %.0f +/- %.0f   top10: %.3f +/- %.3f   max: %.4f +/- %.4f'
      % (gPR.mean(), gPR.std(), gED.mean(), gED.std(), gT10.mean(), gT10.std(), gMX.mean(), gMX.std()))

    # --- 向量级统计 ---
    w('')
    w('[3] 向量级有效维度（组中位数 [q25,q75]）')
    w('    组         n   范数            PR               熵维            top10份额        max份额')
    vec_stats = {}
    for g, E in data.items():
        st = [per_vector(v) for v in E]
        vec_stats[g] = st
        def q(k, st=st):
            a = np.array([s[k] for s in st])
            return np.median(a), np.quantile(a, 0.25), np.quantile(a, 0.75)
        p1, a1, b1 = q('norm')
        p2, a2, b2 = q('PR')
        p3, a3, b3 = q('ED')
        p4, a4, b4 = q('top10')
        p5, a5, b5 = q('mx')
        w('    %-9s %3d  %6.1f [%5.1f,%6.1f]  %6.0f [%5.0f,%6.0f]  %5.0f [%4.0f,%5.0f]  %.3f [%.3f,%.3f]  %.4f [%.4f,%.4f]'
          % (g, len(st), p1, a1, b1, p2, a2, b2, p3, a3, b3, p4, a4, b4, p5, a5, b5))

    w('')
    w('[3b] 用户点名词的逐条读数')
    for gname, pool in (('HIGH', groups_single['HIGH']), ('RARE', groups_single['RARE']), ('RARE2', groups_two['RARE2'])):
        for xw in FOCUS:
            if xw in pool:
                tid = pool[xw]
                st = per_vector(rows([tid])[0])
                w('    %-4s id=%-7d 组=%-5s 范数=%7.1f  PR=%5.0f  熵维=%5.0f  top10=%.3f  max=%.4f'
                  % (xw, tid, gname, st['norm'], st['PR'], st['ED'], st['top10'], st['mx']))

    # --- C1: Mann-Whitney ---
    w('')
    w('[4] C1 判据：HIGH vs RARE（向量级）')
    if 'HIGH' in data and 'RARE' in data:
        for k, lab in (('PR', 'PR'), ('ED', '熵维')):
            a = np.array([s[k] for s in vec_stats['HIGH']])
            b = np.array([s[k] for s in vec_stats['RARE']])
            U, z, p, rbc = mannwhitney(a, b)
            rel = (np.median(a) - np.median(b)) / (np.median(b) + 1e-12)
            verdict = '成立' if (abs(rel) >= 0.10 and p < 0.01) else '不成立'
            w('    %-4s HIGH中位=%.0f RARE中位=%.0f 相对差=%+.1f%%  U=%.0f z=%.2f p=%.2e rankbiserial=%+.2f  => 关系%s（方向：HIGH %s RARE）'
              % (lab, np.median(a), np.median(b), rel * 100, U, z, p, rbc, verdict,
                 '高于' if rel > 0 else '低于'))
    else:
        w('    （HIGH 或 RARE 组存活不足 4，C1 跳过）')

    # --- C3: RARE vs 高斯带 ---
    w('')
    w('[5] C3 判据：RARE 是否 = 未训练随机向量')
    if 'RARE' in data:
        for k, arr, gm, gsd in (('PR', vec_stats['RARE'], float(gPR.mean()), float(gPR.std())),
                                ('ED', vec_stats['RARE'], float(gED.mean()), float(gED.std())),
                                ('top10', vec_stats['RARE'], float(gT10.mean()), float(gT10.std())),
                                ('mx', vec_stats['RARE'], float(gMX.mean()), float(gMX.std()))):
            v = np.array([s[k] for s in arr])
            lo, hi = gm - 2 * gsd, gm + 2 * gsd
            inside = float(((v >= lo) & (v <= hi)).mean())
            w('    %-5s RARE中位=%.4g  高斯带[%.4g, %.4g]  组内落在带内比例=%.2f' % (k, np.median(v), lo, hi, inside))
    else:
        w('    （RARE 组存活不足，C3 跳过）')
    for g in ('HIGH', 'MID'):
        if g in vec_stats:
            v = np.array([s['PR'] for s in vec_stats[g]])
            w('    （对照）%-4s PR中位=%.0f  vs 高斯带 PR %.0f+/-%.0f' % (g, np.median(v), gPR.mean(), gPR.std()))

    # --- B: 组级矩阵有效秩 ---
    w('')
    w('[6] C2 判据：组级矩阵有效秩（K=20 子采样 x50，centered 口径）')
    if 'HIGH' in data and 'RARE' in data:
        Ksub = 20
        res = {}
        for g, E in data.items():
            vals_pr, vals_ent = [], []
            for it in range(50):
                if E.shape[0] > Ksub:
                    idx = rng.choice(E.shape[0], size=Ksub, replace=False)
                    Es = E[idx]
                else:
                    Es = E
                mc = Es.mean(0, keepdims=True)
                s = np.linalg.svd(Es - mc, compute_uv=False)
                e = s * s
                vals_pr.append((e.sum() ** 2) / (np.sum(e * e) + 1e-300))
                qq = e / (e.sum() + 1e-300)
                vals_ent.append(float(np.exp(-np.sum(qq * np.log(qq + 1e-300)))))
            res[g] = (np.median(vals_pr), np.median(vals_ent))
            w('    %-9s centered PR_spec 中位=%6.1f   熵秩中位=%5.1f   (K=%d)' % (g, res[g][0], res[g][1], min(Ksub, E.shape[0])))
        hi, ra = res['HIGH'][0], res['RARE'][0]
        ratio = hi / (ra + 1e-12)
        if ratio < 0.7:
            v2 = 'HIGH < 0.7xRARE => 常用词占更低维子空间（"常用词满秩"否证）'
        elif ratio > 1.3:
            v2 = 'HIGH > 1.3xRARE => 反向'
        else:
            v2 = '组级无差异（0.7-1.3 区间）'
        w('    HIGH/RARE PR_spec 比 = %.2f  => %s' % (ratio, v2))
    else:
        w('    （HIGH 或 RARE 组存活不足，C2 跳过）')

    # --- 全组组指标 ---
    w('')
    w('[7] 全组矩阵指标（不子采样）')
    w('    组         K    raw谱PR  raw熵秩  cen谱PR  cen熵秩  numrank(1e-2/1e-1)  组内cos(raw/centered)  最近邻cos')
    for g, E in data.items():
        gm2 = group_metrics(E)
        w('    %-9s %3d  %7.1f  %7.1f  %7.1f  %7.1f  %4d/%3d  %+.3f / %+.3f  %+.3f'
          % (g, E.shape[0], gm2['raw']['PR_spec'], gm2['raw']['EntRank'],
             gm2['centered']['PR_spec'], gm2['centered']['EntRank'],
             gm2['centered']['num_rank_1e2'], gm2['centered']['num_rank_1e1'],
             gm2['pair_raw']['mean'], gm2['pair_centered']['mean'], gm2['nn_cos']))

    w('')
    w('用时 %.1fs' % (time.time() - t_start))
    with open(rep_path, 'w', encoding='utf-8') as fh:
        fh.write('\n'.join(L))
    print('WROTE', rep_path, len(L), 'lines')

print('ALL DONE %.1fs' % (time.time() - t_start))
