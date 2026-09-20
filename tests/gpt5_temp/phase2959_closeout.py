# -*- coding: utf-8 -*-
"""Phase 2959 closeout: seal + Ledger + MEMO + worklog + MEMORY.
Idempotent, placeholder replacement."""
import hashlib
import json
import os

BASE = (r'D:\AI2050\Ai2050-OpenOne\tests\glm5\result'
        r'\rdc_query_construction_20260913')
OUT = os.path.join(BASE, 'phase2959', 'cross_term_algebra')
LEDGER = (r'D:\AI2050\Ai2050-OpenOne\research\gpt5\atlas'
          r'\atlas_ledger.json')
MEMO = (r'D:\AI2050\Ai2050-OpenOne\research\gpt5\docs'
        r'\AGI_GPT5_MEMO.md')
WLOG = (r'C:\Users\Admin\WorkBuddy\2026-09-17-01-30-05'
        r'\.workbuddy\memory\2026-09-20.md')
WMEM = (r'C:\Users\Admin\WorkBuddy\2026-09-17-01-30-05'
        r'\.workbuddy\memory\MEMORY.md')
SCRIPT = (r'D:\AI2050\Ai2050-OpenOne\tests\glm5'
          r'\phase2959_cross_term_algebra.py')

res = json.load(open(os.path.join(OUT, 'result.json'),
                     encoding='utf-8'))
verdict = res['final_verdict']
created = json.load(open(os.path.join(OUT, 'execution.json'),
                         encoding='utf-8'))['created']


def sha8(path):
    return hashlib.sha256(
        open(path, 'rb').read()).hexdigest()[:8]


npz_name = 'cross_term_algebra.npz'
hashes = {
    'execution.json': sha8(os.path.join(OUT, 'execution.json')),
    'result.json': sha8(os.path.join(OUT, 'result.json')),
    npz_name: sha8(os.path.join(OUT, npz_name)),
    'script': sha8(SCRIPT),
}

# ---------------- Ledger ----------------
led = json.load(open(LEDGER, encoding='utf-8'))
meas_id = 'meas2959_cross_term_algebra'
if not any(m['meas_id'] == meas_id
           for m in led['measurements']):
    led['measurements'].append({
        'meas_id': meas_id,
        'type': 'forward_family_second_order_algebra',
        'verdict': verdict,
        'source': ('phase2959/cross_term_algebra; the dominant '
                   'dq.ddk cross term (2955) is direction-locked '
                   '(cos(dq) 0.992, cos(ddk) 0.994 across doses) '
                   'but NOT second-order in magnitude: log-log '
                   'slope 1.176 (IQR 1.056-1.265, R2 0.970, all '
                   '32 heads) - RMSNorm saturation makes the '
                   'cross term near-linear at large s; s^2 '
                   'small-dose extrapolation overshoots (rel err '
                   'median 1.365) while the per-head PATTERN is '
                   'predictable (rho 0.9685); xt tracks dz (rho '
                   '0.842) but not ATT (rho -0.10)'),
    })
    l14 = [lk for lk in led['linkage']
           if lk['link_id'] == 'L14_readout_spectrum_cross_model'][0]
    if meas_id not in l14['connects']:
        l14['connects'].append(meas_id)
led.pop('ledger_sha256_8', None)
new_h = hashlib.sha256(json.dumps(
    led, sort_keys=True, ensure_ascii=False)
    .encode('utf-8')).hexdigest()[:8]
led['ledger_sha256_8'] = new_h
json.dump(led, open(LEDGER, 'w', encoding='utf-8'),
          indent=2, ensure_ascii=False)
chk = new_h

# ---------------- MEMO ----------------
memo = open(MEMO, encoding='utf-8').read()
if '## Phase 2959' not in memo:
    section = '''## Phase 2959: 交叉项代数结构判决 [@STAMP@]

**判决：`@VERDICT@`** —— 2955 发现的 Δq·Δk 交叉项主导（X=3.88/4.06）**方向可预测但幅度不可由二阶代数外推**：响应方向跨剂量完全锁定（cos(Δq) 中位 0.9918 / cos(Δdk) 0.9944，最差头仍 0.980/0.985），但幅度标度是**近线性而非平方**（log-log 斜率中位 1.176，IQR 1.056–1.265，R²=0.970，全部 32 头入统计）——s² 外推过冲（相对误差中位 1.365），而逐头**模式**高度可预测（spearman 0.9685）。

**设计（2955 verbatim 协议 + s 网格，K=9，runtime @RT@s）**：base×2 + L17 s∈{0.25,0.5,0.75,1.0,1.5,2.0} + L16@s2.0；fp64 q/k 重算链（q_norm/k_norm/RoPE + 1/√HD），精确分解 dz = dq·dk_b + q_b·ddk + dq·ddk。判据冻结：T1 斜率∈[1.7,2.3]→second_order；T2 双 cos≥0.95→direction_locked；T3 rel<0.3 且 rho≥0.9→predictable_small_dose。

**锚 12/12（一次通过）**：a1 dirs 重建 2.17e-08（**第 30 次连续前向锚定**）、a3/a10/a15/a11/a13/a17/**a-iso 全 bit 0**、a7 9.95e-14、a16 dA_med max 8.59e-04（9 条件全 <1e-3）、a18 恒等式 2.49e-14、**a17 vs 2955 npz bit 级 0**（dz/qt/kt/xt/zb/A11sm_b 八键，2957 跨相位锚规范首战全胜）。

**主检验**：
| 检验 | 冻结阈 | 实测 | 判定 |
|---|---|---|---|
| T1 二阶标度 | 斜率∈[1.7,2.3] | 中位 **1.176**（R² 0.970） | anomalous_slope |
| T2 方向锁定 | 双 cos≥0.95 | 0.9918 / 0.9944 | direction_locked |
| T3 小剂量外推 | rel<0.3 & rho≥0.9 | rel **1.365** / rho **0.9685** | not_predictable |

**关键发现**：
1. **交叉项是"方向锁定 + 幅度饱和"的秩 1 响应**：xt 逐头剖面随 s 单调爬升趋缓（如 h22：2.92→9.16→14.6→19.3→23.8→26.7），斜率 1.18 ≈ RMSNorm 归一化的饱和几何（|Δq| 随 s 有界）——二阶代数在小剂量成立、大剂量失效。
2. **操作性含义（原语卡片）**：交叉项可用**两点标定**（方向由任意小剂量 probe 给出、幅度需两点插值），单点 s² 外推禁用；逐头模式 rho 0.97 意味着头级排序跨剂量稳定——图谱的头级签名是剂量不变的。
3. **再解耦**：xt 追踪 dz（rho 0.842）但不追踪 ATT（rho −0.10）——2955 D1 的"logit 位移不预测 ATT 承载"在交叉项层面复现；注意力路由增益仍非单一 logit 标量可解释。
4. sep 全网格单调穿越（182.6→145.9→72.4→21.5→−4.0→−6.4），L16@s2.0 轴 X=4.06 与 2955 复现一致。

**结论（重复 3 次）**：**Δq·Δk 交叉项方向锁定（cos≈0.99）但幅度非二阶（斜率 1.18 饱和）——s² 小剂量外推不成立（rel 1.37），仅逐头模式可预测（rho 0.97）；交叉项的操作化 = 方向一点 + 幅度两点标定。**

**硬伤与勘误（run1→run2）**：run1 **判决记账 bug**——锚全过（12/12）且 T1/T2/T3 已算出，但 verdict 组合行被 `if verdict is None` 兜底吞掉（anchor 分支漏赋值），误标 anchor_fail_all_void。修复为直接按 anchor_prelim 分支 + correction_note，删产物重跑（同 session 确定性，run2 权威）。**教训入 MEMORY**：verdict 必须在判据分支内显式赋值，兜底分支只允许留给锚失败路径。

**文件+SHA256-8**：execution @HEXE@ / result @HRES@ / npz @HNPZ@ / script @HSCR@。产物 `tests/glm5/result/rdc_query_construction_20260913/phase2959/cross_term_algebra/`。Ledger 98 条 / L14 connects 66 / ledger @LEDHASH@。

**接续**：机制链第二十二环（交叉项代数环）闭合，方案 v2 阶段一过半。候选 2960：A（主选）剖面旋转定位（2958 遗留：L17 消融 MLP 剖面随 k 旋转的几何——旋转轴与固定分量分解，一次前向族）；B 原语卡片压缩（把 21 环机制链压缩为层带×模块×头集×读出×剂量律×lin_r 的结构化卡片表，纯文档 Phase）；C 词类机制签名矩阵预研（阶段二启动：词表扩容 n≳40 分组设计）；D v3 解码器方向重启（2940 遗留）。
'''
    stamp = created.replace('T', ' ')[:16]
    reps = [('@STAMP@', stamp), ('@VERDICT@', verdict),
            ('@RT@', str(res['runtime_s'])),
            ('@LEDHASH@', chk),
            ('@HEXE@', hashes['execution.json']),
            ('@HRES@', hashes['result.json']),
            ('@HNPZ@', hashes[npz_name]),
            ('@HSCR@', hashes['script'])]
    for k, v in reps:
        section = section.replace(k, v)
    memo += section
    open(MEMO, 'w', encoding='utf-8').write(memo)

# ---------------- worklog (create if missing) ----------------
entry = ('- Phase 2959 闭环：anomalous_slope_direction_locked_'
         'not_predictable。2955 的 Δq·Δk 交叉项主导项：方向跨剂量'
         '锁定（cos 0.992/0.994）但幅度非二阶（log-log 斜率 1.176，'
         'R² 0.970，RMSNorm 饱和）——s² 小剂量外推过冲（rel 1.365）'
         '而逐头模式可预测（rho 0.9685）；操作性结论 = 方向一点 + '
         '幅度两点标定。xt 追踪 dz（0.842）不追踪 ATT（−0.10）再'
         '解耦。run1 判决记账 bug（锚全过但 verdict 被兜底吞掉）→ '
         'run2 权威；12/12 锚（a17 vs 2955 bit 0 八键；a1 第 30 次'
         '连续前向锚定）。Ledger 98 / hash ' + chk + '。\n')
wl = ''
if os.path.exists(WLOG):
    wl = open(WLOG, encoding='utf-8').read()
if 'Phase 2959' not in wl:
    wl += entry
    open(WLOG, 'w', encoding='utf-8').write(wl)

# ---------------- MEMORY ----------------
mem = open(WMEM, encoding='utf-8').read()
mem = mem.replace(
    '当前 max=**2958**，下一个 **2959**（方案 v2 阶段一首战）',
    '当前 max=**2959**，下一个 **2960**（候选 A 剖面旋转定位）')
old = '→驱动律：印记剂量单调无阈值，层类型定几何模式(2958)。'
if old in mem:
    mem = mem.replace(old, old[:-1]
                      + '→交叉项代数：方向锁定幅度饱和(2959)。')
if '## 机制链状态（21 环）' in mem:
    mem = mem.replace('## 机制链状态（21 环）',
                      '## 机制链状态（22 环）')
if '连续 29 次前向锚定' in mem:
    mem = mem.replace('连续 29 次前向锚定',
                      '连续 30 次前向锚定')
add = ('- 判决记账规范（2959）：verdict 必须在判据分支内显式赋值，'
       '"if verdict is None" 兜底只允许留给锚失败路径——否则锚全过'
       '也会被误标 anchor_fail_all_void；二阶代数外推禁令（2959）：'
       'RMSNorm 归一化下 q/k 响应方向跨剂量锁定（cos≈0.99）但幅度'
       '饱和（log-log 斜率 1.18 非 2），单点 s² 外推禁用，交叉项'
       '操作化 = 方向一点 + 幅度两点标定（头级排序 rho 0.97 剂量'
       '不变）。\n')
if '判决记账规范（2959）' not in mem:
    anchor_line = '- 跨相位 bit 级锚规范（2957）：'
    i = mem.find(anchor_line)
    if i >= 0:
        mem = mem[:i] + add + mem[i:]
    else:
        mem = mem.rstrip() + '\n' + add
open(WMEM, 'w', encoding='utf-8').write(mem)

print('closeout done: ledger=%s n_meas=%s' % (
    chk, len(led['measurements'])))
