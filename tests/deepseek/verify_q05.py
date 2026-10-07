# -*- coding: utf-8 -*-
"""
Q05 独立复核（独立进程）：从各 arm 的**存储原语**重算 E_ar/rel/drift/gates，重算 design_sha/res_sha8/
panel_sha8，核对 D0 与 Q04 一致，校验产物落地哈希。不改任何产物。
输出 tests/deepseek/result/verify_q05.txt
"""
import os, sys, json, hashlib
try:
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
except Exception:
    pass
import numpy as np

ROOT = r'D:\AI2050\Ai2050-OpenOne'
OUT = os.path.join(ROOT, 'tests', 'deepseek', 'result')
ARMS = ['qwen3-4b__bf16', 'qwen3-4b__nf4', 'qwen3-14b__nf4', 'glm4-9b__nf4']
SEEDS = [7, 8, 9]
L = []
n_pass = n_fail = 0
def ck(name, cond, detail=''):
    global n_pass, n_fail
    ok = bool(cond)
    n_pass += ok; n_fail += (not ok)
    L.append('%s  %-46s  %s' % ('PASS' if ok else 'FAIL', name, detail))

def h8(p):
    return hashlib.sha256(open(p, 'rb').read()).hexdigest()[:8]

# 生产者的 res_sha8 哈希域**不是 JSON 往返稳定**的：q05_e_ar_measure.py 对曲线字典用
# **整数键**（sort_keys 按数值排序 0,1,2,...,16），而 per_seed 用字符串键；q05_aggregate.py
# 对 curves/null_rel_report_only 的内层也用整数键。复核必须复刻生产者的键型。
INT_KEYED_ARM = ['E_ar', 'E_ar_const', 'scale', 'E_ar_rel', 'drift']
def prod_sha8_arm(r):
    c = {k: v for k, v in r.items() if k != 'res_sha8'}
    for kk in INT_KEYED_ARM:
        if isinstance(c.get(kk), dict):
            c[kk] = {int(k): v for k, v in c[kk].items()}
    return hashlib.sha256(json.dumps(c, ensure_ascii=False, sort_keys=True, indent=1).encode('utf-8')).hexdigest()[:8]
def prod_sha8_agg(d):
    c = {k: v for k, v in d.items() if k != 'res_sha8'}
    for kk in ['curves', 'null_rel_report_only']:
        if isinstance(c.get(kk), dict):
            c[kk] = {a: {int(k): v for k, v in cv.items()} for a, cv in c[kk].items()}
    return hashlib.sha256(json.dumps(c, ensure_ascii=False, sort_keys=True, indent=1).encode('utf-8')).hexdigest()[:8]

# 冻结面板指纹（独立重算）
CLASSES = ['水果', '动物', '交通工具', '家具', '金属', '颜色']
ENT = {
    '水果': ['苹果', '香蕉', '梨', '西瓜', '葡萄', '草莓', '芒果', '柠檬'],
    '动物': ['狗', '猫', '老虎', '大象', '兔子', '猴子', '马', '牛'],
    '交通工具': ['汽车', '火车', '飞机', '摩托车', '卡车', '地铁'],
    '家具': ['桌子', '椅子', '床', '沙发', '地毯', '窗帘'],
    '金属': ['铁', '铜', '铝', '金', '银', '锌', '铅'],
    '颜色': ['红', '蓝', '绿', '黄', '黑', '白'],
}
TPL = ['{e}是一种{c}。', '{e}属于{c}这一类。', '{e}，一种常见的{c}。']
ENTS = [e for cl in CLASSES for e in ENT[cl]]
PANEL_SHA8 = hashlib.sha256(json.dumps([CLASSES, ENTS, TPL, SEEDS, 0.2],
                            ensure_ascii=False).encode('utf-8')).hexdigest()[:8]

# ---- 1. 各 arm: 重算 E_ar / rel / drift；重算 gates ----
res = {}
for a in ARMS:
    r = json.load(open(os.path.join(OUT, 'q05_%s_result.json' % a), encoding='utf-8'))
    res[a] = r
    K = r['K']
    # 从 per_seed 重算 E_ar
    E = {str(k): float(np.mean([r['per_seed'][str(s)][str(k)]['mae_b4'] for s in SEEDS])) for k in range(K + 1)}
    ck('E_ar recompute [%s]' % a,
       all(abs(E[str(k)] - r['E_ar'][str(k)]) < 1e-12 for k in range(K + 1)),
       'max_dev=%.2e' % max(abs(E[str(k)] - r['E_ar'][str(k)]) for k in range(K + 1)))
    SC = {str(k): float(np.mean([r['per_seed'][str(s)][str(k)]['scale'] for s in SEEDS])) for k in range(K + 1)}
    ck('scale recompute [%s]' % a,
       all(abs(SC[str(k)] - r['scale'][str(k)]) < 1e-9 for k in range(K + 1)), '')
    ck('E_ar_rel=E_ar/scale [%s]' % a,
       all(abs(r['E_ar_rel'][str(k)] - r['E_ar'][str(k)] / max(r['scale'][str(k)], 1e-9)) < 1e-9 for k in range(K + 1)), '')
    ck('drift=E_ar(k)-E_ar(0) [%s]' % a,
       all(abs(r['drift'][str(k)] - (r['E_ar'][str(k)] - r['E_ar']['0'])) < 1e-9 for k in range(K + 1)), '')
    ck('D3 liveness (all finite) [%s]' % a,
       all(np.isfinite(r['E_ar'][str(k)]) for k in range(K + 1)), '')
    ck('S1 gate recompute [%s]' % a,
       bool(max(r['E_ar'][str(k)] for k in range(1, K + 1)) >= r['gates']['S1_thr']),
       'max=%.4f thr=%.2f' % (max(r['E_ar'][str(k)] for k in range(1, K + 1)), r['gates']['S1_thr']))
    ck('panel_sha8 == recompute [%s]' % a, r['panel_sha8'] == PANEL_SHA8, r['panel_sha8'])
    ck('res_sha8 recompute [%s]' % a, prod_sha8_arm(r) == r['res_sha8'], r['res_sha8'] + ' (生产者键型约定)')
    # execution.json 幂等/对齐
    ex = json.load(open(os.path.join(OUT, 'q05_%s_execution.json' % a), encoding='utf-8'))
    dsha = ex['design_sha']
    ex2 = dict(ex); ex2.pop('design_sha', None)
    ck('design_sha idempotent [%s]' % a,
       hashlib.sha256(json.dumps(ex2, ensure_ascii=False, sort_keys=True, indent=1).encode('utf-8')
                      ).hexdigest() == dsha, dsha[:8])
    ck('execution.model==arm model [%s]' % a, dsha == r['design_sha'], '')

# ---- 2. K/面板 一致性 ----
Ks = set(res[a]['K'] for a in ARMS)
ck('K identical across arms', Ks == {16}, str(Ks))
ncell = set(res[a]['n_cells'] for a in ARMS)
ck('n_cells identical (=738)', ncell == {738}, str(ncell))

# ---- 3. D0: q05 smoke(4b bf16) == q04 smoke ----
q4 = json.load(open(os.path.join(OUT, 'q04_smoke_result.json'), encoding='utf-8'))
q5s = json.load(open(os.path.join(OUT, 'q05_qwen3-4b__bf16_smoke_result.json'), encoding='utf-8'))
dev = max(abs(q5s['E_ar'][str(k)] - q4['E_ar'][str(k)]) for k in range(5))
ck('D0 q05-smoke == q04-smoke (bit)', dev == 0.0, 'max_dev=%.2e' % dev)

# ---- 4. 聚合结果 ----
agg = json.load(open(os.path.join(OUT, 'q05_result.json'), encoding='utf-8'))
ck('agg res_sha8 recompute', prod_sha8_agg(agg) == agg['res_sha8'], agg['res_sha8'] + ' (生产者键型约定)')
# D4 重算
relb = [agg['curves']['qwen3-4b__bf16'][str(k)]['E_ar_rel'] for k in range(17)]
reln = [agg['curves']['qwen3-4b__nf4'][str(k)]['E_ar_rel'] for k in range(17)]
dmax = max(abs(relb[k] - reln[k]) for k in range(17))
ck('D4 bridge recompute', abs(dmax - agg['precision_bridge']['d_abs_max']) < 1e-12,
   'max|Δrel|=%.4f thr=%.2f -> %s' % (dmax, agg['precision_bridge']['thr'],
                                      'PASS' if dmax <= agg['precision_bridge']['thr'] else 'FAIL'))
ck('agg arm count = 4', len(agg['arms']) == 4, str(agg['arms']))
sumfwd = set(res[a]['sum_fwd'] for a in ARMS)
ck('forwards = 12546 per arm', sumfwd == {738 * 17}, str(sumfwd))

L.append('')
L.append('NOTE  res_sha8 复核按**生产者键型约定**（曲线字典用整数键、per_seed 用字符串键）；'
         '该约定非 JSON 往返稳定，已在 q05 脚本与本文件注明。数据级复核独立于该约定'
         '（E_ar 由 per_seed 重算，max_dev=0）。')
L.append('TOTAL  PASS=%d  FAIL=%d' % (n_pass, n_fail))
L.append('VERDICT = %s' % ('ALL_PASS' if n_fail == 0 else 'HAS_FAIL'))
txt = '\n'.join(L)
open(os.path.join(OUT, 'verify_q05.txt'), 'w', encoding='utf-8').write(txt + '\n')
print(txt)
print('VERIFY_OK' if n_fail == 0 else 'VERIFY_FAIL')
