# -*- coding: utf-8 -*-
"""Phase 3156 addendum: KL_B / top1_B / norm-collapse spectrum / per-layer rope
Reads sealed collect.npz only; writes result_addendum.json (independent artifact)."""
import os, json, hashlib, time
import numpy as np

T0 = time.time()
BASE = r'D:\AI2050\Ai2050-OpenOne\tests\glm5\result\rdc_query_construction_20260913\phase3156\g3p1_position_shift_family\qwen3-4b'
z = np.load(os.path.join(BASE, 'collect.npz'))
H = z['H'].astype(np.float32)
LG = z['LG'].astype(np.float32)
langs = [s.decode() if isinstance(s, bytes) else str(s) for s in z['lang']]
arms = [s.decode() if isinstance(s, bytes) else str(s) for s in z['arm']]
ks = [int(v) for v in z['k']]
ntg = [int(v) for v in z['n_tgt']]
IDX = {(langs[i], arms[i], ks[i]): i for i in range(len(langs))}
KG = sorted(set(ks))
NH = H.shape[2]
NL = NH - 1

def logsoftmax(v):
    v = v - v.max()
    return v - np.log(np.exp(v).sum())

out = dict(phase=3156, kind='addendum', model='qwen3-4b',
           note='position-effect vs context-effect separation from sealed npz; runtime %.1fs' % (time.time() - T0))

# 1) B 臂输出恢复: KL(B_k || A0), top1_B
kl_b, top_b = {}, {}
for lg in set(langs):
    ia0 = IDX[(lg, 'A', 0)]
    p0 = np.exp(logsoftmax(LG[ia0]))
    for k in KG:
        ib = IDX[(lg, 'B', k)]
        pk = np.exp(logsoftmax(LG[ib]))
        kl_b['%s_k%d' % (lg, k)] = float((p0 * (np.log(p0 + 1e-30) - np.log(pk + 1e-30))).sum())
        top_b['%s_k%d' % (lg, k)] = float(int(np.argmax(LG[ib])) == int(np.argmax(LG[ia0])))
out['kl_B'] = kl_b
out['top1_B'] = top_b

# 2) 范数塌缩谱: ||A_k(l)|| / ||A0(l)|| 与 maxabs, per lang
norm_spec = {}
for lg in set(langs):
    ia0 = IDX[(lg, 'A', 0)]
    n = ntg[ia0]
    for k in KG:
        ia = IDX[(lg, 'A', k)]
        ratio = [float(np.linalg.norm(H[ia, :n, l]) / (np.linalg.norm(H[ia0, :n, l]) + 1e-18))
                 for l in range(NH)]
        mx0 = float(np.abs(H[ia0, :n]).max())
        mxk = float(np.abs(H[ia, :n]).max())
        norm_spec['%s_k%d' % (lg, k)] = dict(norm_ratio=[round(x, 4) for x in ratio],
                                             maxabs_A0=round(mx0, 1), maxabs_Ak=round(mxk, 1))
out['norm_collapse_A'] = norm_spec

# 3) 逐层 rope rel: ||B_k(l)-A0(l)||/||A0(l)||
rope_l = {}
for lg in set(langs):
    ia0 = IDX[(lg, 'A', 0)]
    n = ntg[ia0]
    for k in KG:
        if k == 0:
            continue
        ib = IDX[(lg, 'B', k)]
        rope_l['%s_k%d' % (lg, k)] = [float(np.linalg.norm((H[ia0, :n, l] - H[ib, :n, l]).ravel()) /
                                            (np.linalg.norm(H[ia0, :n, l].ravel()) + 1e-18))
                                      for l in range(NH)]
out['rope_per_layer'] = {kk: [round(x, 6) for x in v] for kk, v in rope_l.items()}

# 4) (merged into rope_per_layer - same quantity)

# 5) A 臂位移方向低秩性复核(readout + mid, k=128, zh/en)
sv_out = {}
for lg in set(langs):
    ia0, ia = IDX[(lg, 'A', 0)], IDX[(lg, 'A', 128)]
    n = ntg[ia0]
    for l in (NL // 2, NL):
        dlt = H[ia, :n, l] - H[ia0, :n, l]
        s = np.linalg.svd(dlt, compute_uv=False)
        e = s ** 2
        sv_out['%s_L%d' % (lg, l)] = dict(top1=round(float(e[0] / e.sum()), 4),
                                          top2=round(float(e[:2].sum() / e.sum()), 4),
                                          top4=round(float(e[:4].sum() / e.sum()), 4))
out['ctx_svd'] = sv_out

blob = json.dumps(out, ensure_ascii=False, indent=1, sort_keys=True).encode('utf-8')
sha8 = hashlib.sha256(blob).hexdigest()[:8]
out['addendum_sha8'] = sha8
with open(os.path.join(BASE, 'result_addendum.json'), 'w', encoding='utf-8') as f:
    json.dump(out, f, ensure_ascii=False, indent=1, sort_keys=True)

lines = ['addendum sha8=%s' % sha8]
lines.append('KL_B per k: %s' % {k.split('_k')[1]: round(v, 4) for k, v in sorted(kl_b.items(), key=lambda x: int(x[0].split('_k')[1]))})
lines.append('top1_B: %s' % {k.split('_k')[1]: round(v, 2) for k, v in sorted(top_b.items(), key=lambda x: int(x[0].split('_k')[1]))})
lines.append('norm_ratio zh k=128 every 6th layer: %s' % norm_spec['zh_k128']['norm_ratio'][::6])
lines.append('norm_ratio zh k=1: %s' % norm_spec['zh_k1']['norm_ratio'][::6])
lines.append('maxabs: A0=%.1f A128=%.1f | en A0=%.1f A128=%.1f' % (
    norm_spec['zh_k128']['maxabs_A0'], norm_spec['zh_k128']['maxabs_Ak'],
    norm_spec['en_k128']['maxabs_A0'], norm_spec['en_k128']['maxabs_Ak']))
lines.append('rope per-layer zh k=128 max=%.2e argmaxL=%d' % (
    max(rope_l['zh_k128']), int(np.argmax(rope_l['zh_k128']))))
lines.append('rope per-layer en k=128 max=%.2e argmaxL=%d' % (
    max(rope_l['en_k128']), int(np.argmax(rope_l['en_k128']))))
lines.append('rope per-layer en k=64 max=%.2e argmaxL=%d' % (
    max(rope_l['en_k64']), int(np.argmax(rope_l['en_k64']))))
lines.append('ctx_svd: %s' % sv_out)
with open(r'D:\AI2050\Ai2050-OpenOne\tests\gpt5_temp\p3156_addendum_out.txt', 'w', encoding='utf-8') as f:
    f.write(chr(10).join(lines))
print('written')
