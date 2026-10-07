# -*- coding: utf-8 -*-
"""
Q05 聚合: 读取四个 arm 结果 -> D4 精度桥 + 形状/半衰期判决 + S_rel -> q05_result.json / q05_report.txt
预注册读自 tests/deepseek/result/q05_prereg_bridge_v1.json（在 nf4 观测前冻结）。
"""
import os, sys, json, hashlib
try:
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
except Exception:
    pass
import numpy as np

ROOT = r'D:\AI2050\Ai2050-OpenOne'
OUT = os.path.join(ROOT, 'tests', 'deepseek', 'result')
PRE = json.load(open(os.path.join(OUT, 'q05_prereg_bridge_v1.json'), encoding='utf-8'))
ARMS = ['qwen3-4b__bf16', 'qwen3-4b__nf4', 'qwen3-14b__nf4', 'glm4-9b__nf4']

R = {a: json.load(open(os.path.join(OUT, 'q05_%s_result.json' % a), encoding='utf-8')) for a in ARMS}
K = R['qwen3-4b__bf16']['K']
assert all(R[a]['K'] == K for a in ARMS), 'K mismatch across arms'
assert all(R[a]['n_cells'] == R['qwen3-4b__bf16']['n_cells'] for a in ARMS), 'panel size mismatch'
P_SHA = R['qwen3-4b__bf16']['panel_sha8']
assert all(R[a]['panel_sha8'] == P_SHA for a in ARMS), 'panel_sha8 mismatch'

def curve(a, key):
    return [R[a][key][str(k)] for k in range(K + 1)]

def classify_shape(a):
    E = curve(a, 'E_ar'); S = curve(a, 'scale')
    g = [E[k] - E[0] for k in range(K + 1)]
    G = g[K]
    tol = 0.05 * max(S[K], 1e-9)
    if G <= tol:
        return dict(shape='flat', G=G, tol=tol, half_life_k=None, mono=None, m2=None)
    hl = next((k for k in range(1, K + 1) if g[k] >= 0.5 * G), None)
    d2 = np.diff(np.array(g), 2)
    m2 = float(np.mean(d2)) if len(d2) else 0.0
    shape = 'linear' if abs(m2) <= 0.02 * abs(G) else ('saturating' if m2 < 0 else 'diverging')
    mono = float(np.mean([1.0 if E[k] >= E[k - 1] else 0.0 for k in range(1, K + 1)]))
    return dict(shape=shape, G=G, tol=tol, half_life_k=hl, mono=mono, m2=m2)

def s_rel(a):
    rel = curve(a, 'E_ar_rel')
    mn = min(rel[k] for k in range(1, K + 1))
    return dict(min_rel_k1_K=mn, pass_=bool(mn <= 0.05), argmin_k=int(np.argmin([rel[k] for k in range(1, K + 1)])) + 1)

# ---- D4 精度桥（qwen3-4b bf16 vs nf4） ----
rel_bf16 = curve('qwen3-4b__bf16', 'E_ar_rel')
rel_nf4 = curve('qwen3-4b__nf4', 'E_ar_rel')
d_abs = [abs(rel_nf4[k] - rel_bf16[k]) for k in range(K + 1)]
d_abs_max = max(d_abs)
thr = PRE['D4_precision_bridge']['THR']
rel_den = max(max(rel_bf16), 1e-9)
d_rel = d_abs_max / rel_den
d4_pass = bool(d_abs_max <= thr)
d4_sec_pass = bool(d_rel <= 0.25)

shapes = {a: classify_shape(a) for a in ARMS}
srels = {a: s_rel(a) for a in ARMS}

verdict = 'Q05_DONE'
verdict += '|D4_BRIDGE_%s' % ('PASS' if d4_pass else 'FAIL')
if not d4_pass:
    verdict += '|NF4_ARMS_DESCRIPTIVE_ONLY'
verdict += '|S_rel_%s(%d/%d arm 过门)' % ('PASS' if all(srels[a]['pass_'] for a in ARMS) else 'FAIL',
                                            sum(1 for a in ARMS if srels[a]['pass_']), len(ARMS))
verdict += '|P4b=%s|P14b=%s|P9b=%s' % (shapes['qwen3-4b__bf16']['shape'],
                                        shapes['qwen3-14b__nf4']['shape'],
                                        shapes['glm4-9b__nf4']['shape'])

result = dict(
    query='Q05', stage='aggregate', K=K, panel_sha8=P_SHA,
    arms=ARMS,
    curves={a: {k: dict(E_ar=R[a]['E_ar'][str(k)], E_ar_rel=R[a]['E_ar_rel'][str(k)],
                        scale=R[a]['scale'][str(k)], E_ar_const=R[a]['E_ar_const'][str(k)],
                        drift=R[a]['drift'][str(k)]) for k in range(K + 1)} for a in ARMS},
    per_arm_verdict={a: R[a]['verdict'] for a in ARMS},
    per_arm_res_sha8={a: R[a]['res_sha8'] for a in ARMS},
    precision_bridge=dict(arm_a='qwen3-4b__bf16', arm_b='qwen3-4b__nf4',
                          d_abs_per_k={str(k): float(d_abs[k]) for k in range(K + 1)},
                          d_abs_max=float(d_abs_max), thr=thr, pass_=d4_pass,
                          d_rel=d_rel, D4_secondary_pass=d4_sec_pass,
                          on_result='nf4 臂与 bf16 可比' if d4_pass else 'nf4 臂降级 descriptive-only'),
    shape={a: shapes[a] for a in ARMS},
    s_rel={a: srels[a] for a in ARMS},
    s_rel_all_pass=bool(all(srels[a]['pass_'] for a in ARMS)),
    null_rel_report_only={a: {k: dict(rel_null=float(R[a]['E_ar_const'][str(k)] / max(R[a]['scale'][str(k)], 1e-9)))
                              for k in range(K + 1)} for a in ARMS},
    verdict=verdict,
)
blob = json.dumps(result, ensure_ascii=False, sort_keys=True, indent=1).encode('utf-8')
result['res_sha8'] = hashlib.sha256(blob).hexdigest()[:8]
json.dump(result, open(os.path.join(OUT, 'q05_result.json'), 'w', encoding='utf-8'),
          ensure_ascii=False, indent=1)

# ---- 报告 ----
T = []
T.append('Q05: E_ar(k) 正式测量 —— 聚合报告')
T.append('=' * 78)
T.append('panel_sha8 = %s   K = %d   res_sha8 = %s' % (P_SHA, K, result['res_sha8']))
T.append('arms = %s' % ', '.join(ARMS))
T.append('')
T.append('每 arm 的 E_ar(k)（原始 L1, logit）与 rel=E_ar/scale:')
hdr = '  %-4s' % 'k' + ''.join('%-24s' % a for a in ARMS)
T.append(hdr)
for k in range(K + 1):
    row = '  %-4d' % k
    for a in ARMS:
        row += '%-24s' % ('%.3f (r%.3f)' % (R[a]['E_ar'][str(k)], R[a]['E_ar_rel'][str(k)]))
    T.append(row)
T.append('')
T.append('D4 精度桥（qwen3-4b: bf16 vs nf4）:')
T.append('  max_k |Δrel| = %.4f   (门 %.2f)  -> %s' % (d_abs_max, thr, 'PASS' if d4_pass else 'FAIL'))
T.append('  相对形式 max|Δrel|/max(rel_bf16) = %.4f  (门 0.25) -> %s' % (d_rel, 'PASS' if d4_sec_pass else 'FAIL'))
T.append('  结论: %s' % result['precision_bridge']['on_result'])
T.append('')
T.append('形状判决 + S_rel:')
for a in ARMS:
    sh = shapes[a]; sr = srels[a]
    T.append('  %-16s shape=%-10s G=%+.3f half_life_k=%s mono=%.2f | S_rel min=%.4f %s'
             % (a, sh['shape'], sh['G'], sh['half_life_k'], (sh['mono'] or -1), sr['min_rel_k1_K'],
                'PASS' if sr['pass_'] else 'FAIL'))
T.append('')
T.append('对照（report-only，非门）：rel_null(k) = E_ar_const / scale —— 平凡「训练均值」预测者的相对 L1。')
T.append('  高斯下常数预测者 rel_null ≈ 0.80；B4 须显著低于它才算对该目标有解释力。')
for a in ARMS:
    c = [R[a]['E_ar_const'][str(k)] / max(R[a]['scale'][str(k)], 1e-9) for k in range(K + 1)]
    T.append('  %-16s rel_null: k0=%.3f k16=%.3f mean=%.3f  |  B4 rel 均值=%.3f'
             % (a, c[0], c[K], float(np.mean(c)), float(np.mean([R[a]['E_ar_rel'][str(k)] for k in range(K + 1)]))))
T.append('')
T.append('VERDICT = %s' % verdict)
txt = '\n'.join(T)
open(os.path.join(OUT, 'q05_report.txt'), 'w', encoding='utf-8').write(txt + '\n')
print(txt)
print('AGG_OK')
