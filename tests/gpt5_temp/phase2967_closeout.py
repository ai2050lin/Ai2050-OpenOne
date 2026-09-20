# -*- coding: utf-8 -*-
"""Phase 2967 closeout: seal + Ledger + MEMO + worklog + MEMORY.
Idempotent, placeholder replacement."""
import hashlib
import json
import os

BASE = (r'D:\AI2050\Ai2050-OpenOne\tests\glm5\result'
        r'\rdc_query_construction_20260913')
OUT = os.path.join(BASE, 'phase2967', 'collapse_carrier_anatomy')
LEDGER = (r'D:\AI2050\Ai2050-OpenOne\research\gpt5\atlas'
          r'\atlas_ledger.json')
MEMO = (r'D:\AI2050\Ai2050-OpenOne\research\gpt5\docs'
        r'\AGI_GPT5_MEMO.md')
WLOG = (r'C:\Users\Admin\WorkBuddy\2026-09-17-01-30-05'
        r'\.workbuddy\memory\2026-09-20.md')
WMEM = (r'C:\Users\Admin\WorkBuddy\2026-09-17-01-30-05'
        r'\.workbuddy\memory\MEMORY.md')
SCRIPT = (r'D:\AI2050\Ai2050-OpenOne\tests\glm5'
          r'\phase2967_collapse_carrier_anatomy.py')

res = json.load(open(os.path.join(OUT, 'result.json'),
                     encoding='utf-8'))
verdict = res['final_verdict']
created = json.load(open(os.path.join(OUT, 'execution.json'),
                         encoding='utf-8'))['created']


def sha8(path):
    return hashlib.sha256(
        open(path, 'rb').read()).hexdigest()[:8]


hashes = {
    'execution.json': sha8(os.path.join(OUT, 'execution.json')),
    'result.json': sha8(os.path.join(OUT, 'result.json')),
    'collapse_carrier.npz': sha8(
        os.path.join(OUT, 'collapse_carrier.npz')),
    'script': sha8(SCRIPT),
}

# ---------------- Ledger ----------------
led = json.load(open(LEDGER, encoding='utf-8'))
meas_id = 'meas2967_collapse_carrier_anatomy'
if not any(m['meas_id'] == meas_id
           for m in led['measurements']):
    led['measurements'].append({
        'meas_id': meas_id,
        'type': 'collapse_carrier_anatomy',
        'verdict': verdict,
        'source': ('phase2967/collapse_carrier_anatomy; '
                   '2966-identical forward family, full-layer '
                   'C[36,s,57,32] retained; anchors 14/14 incl '
                   'a12/a13/a14 vs 2966 npz all bit 0.0; T1 '
                   '12 sig layers, 8 carrier layers L{24,26,'
                   '27,28,30,31,32,33} (ratio<0.3), L34 sig '
                   'q 1e-3 but ratio 0.5148; T2 318/1152 sig '
                   'heads, L34 12 heads superset of 2966 top5; '
                   'T1 note: L17 amplifies (ratio 367.5), '
                   'collapse is downstream-deep phenomenon'),
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
if '## Phase 2967' not in memo:
    section = '''## Phase 2967: B(s) 单调塌缩载体解剖——深层带 L24-33 八层承载，L34 显著响应但端点未塌尽 [@STAMP@]

**判决：`@VERDICT@`**（run2 权威）——2966 发现 rho(s,B)=+0.9364（B 被语言注入单调抹平 −1.365→−0.019）后，哪些层/头承载塌缩：2966-identical 前向族（57 词 verbatim、GRID17+s0、K=1），本次保留全层头级贡献 C[36,s,57,32]。

**correction_note**：run1 T2 向量化 spearman 中 `np.where(deg2[:,None], x(1152,), 1.0)` 条件 (1152,1) 与 x (1152,) 静默广播成 (1152,1152) 矩阵致 matmul 维度错——np.where 条件与候选形状不一致的广播陷阱；修 deg2 一维后按纪律 3 删旧产物重跑一次通过。新增工程教训入 MEMORY。

**锚 14/14**：a1-a11 与 2966 逐项一致（a1 2.17e-08 / a2 0.0 / a3 0.0 / a4 7.21e-06 / a5 6.26e-06 / a6 185.6975 / a7 9.95e-14 / a9 1.81e-07 / a10 0.162 / a11 0.0）；**a12 C34_curves / a13 B_curve / a14 sep_curve vs 2966 npz 全部 = 0.0（bit 级同实现复制）**——预检预言兑现（2937→2959 先例）。

**主检验**：
1. **T1 层级 gap 曲线（主检验，36 层 maxT 族 rng 2975×10000）**：12 显著层 [17,20,21,24,26,27,28,30,31,32,33,34]；塌缩载体层（显著 ∧ 端点比 |gap(2)/gap(0)|<0.3）= **L{24,26,27,28,30,31,32,33} 八层**（最强 L33 0.0005 / L31 0.0016 / L30 0.041 / L26 0.076）；**L34 显著（q 1e-3、rho −0.9364）但端点比 0.5148——响应显著而未塌尽**（与 h15 biphasic 峰后回落至 −0.06 一致，类载体层"变浅但存活"）；**L17 自身端点比 367.5（放大而非塌缩）**——塌缩是注入层下游的深层带现象，非注入点局部。
2. **T2 头级族（1152 头 maxT rng 2976×10000，非退化门纪律 12）**：**318/1152 显著**；L34 显著头 12 个 [2,3,6,7,8,10,13,16,21,24,30,31]，**严格包含 2966 描述性 top5 {6,31,24,2,10}——描述性发现获正式复制**。
3. **T3 描述性**：头级 top12 全部 |rho|=1.0 完美单调：(34,6)/(33,31)/(32,18)/(29,18)/(30,1)/(30,20)/(31,31)/(31,26)/(30,15)/(26,3)/(28,24)/(26,12)——深层带存在大量完美跟随注入剂量的头，塌缩由层带级分布式头群承载，无单点。

**结论（重复 3 次）**：**B 的单调塌缩由 L24-33 深层带八层（318 头）承载，是层带级分布式载体而非单点；"语言轴注入 → L17 开关 → 下游 L24-33 深层带 gap 抹平 → B 近零"因果链路闭合，而类载体层 L34 位于该链路下游且只部分塌缩（ratio 0.51）——词类签名在语言信号被抹平时同步变浅但保留残差，与 2966 h15 biphasic 瞬态一致。**

**硬伤与勘误**：run1 T2 np.where 广播 bug（见 correction_note）；数据三次运行 bit 级一致。

**文件+SHA256-8**：execution @HEXE@ / result @HRES@ / collapse_carrier.npz @HNpz@ / script @HSCR@。产物 `tests/glm5/result/rdc_query_construction_20260913/phase2967/collapse_carrier_anatomy/`。Ledger 106 条 / L14 connects 74 / ledger @LEDHASH@。

**接续**：候选 2968：A（主选）**h15 biphasic 峰位表征**（逐词峰位分布 vs s_c、跨注入方向泛化——判定峰锁 s_c 是个体性质还是群体性质）；B L34 残差 signature：塌缩后仍存活的 0.51×gap 的头级构成（2967 npz 已有数据，离线分析）；C S1 路由边界带扩容复检（n≥60 加 L34）；D 2961 卡组扩充（补 2962-2967 七行）。
'''
    stamp = created.replace('T', ' ')[:16]
    reps = [('@STAMP@', stamp), ('@VERDICT@', verdict),
            ('@LEDHASH@', chk),
            ('@HEXE@', hashes['execution.json']),
            ('@HRES@', hashes['result.json']),
            ('@HNpz@', hashes['collapse_carrier.npz']),
            ('@HSCR@', hashes['script'])]
    for k, v in reps:
        section = section.replace(k, v)
    memo += section
    open(MEMO, 'w', encoding='utf-8').write(memo)

# ---------------- worklog ----------------
entry = ('- Phase 2967 闭环：collapse_carrier_localized（run2 权威，锚 14/14，'
         'a12-14 vs 2966 npz 全 bit 0）。B 单调塌缩载体 = L24-33 深层带八层'
         '（318/1152 头显著，top12 全 |rho|=1.0）；L34 显著响应（q 1e-3）但端点比 '
         '0.5148 未塌尽——类载体层"变浅但存活"，与 h15 biphasic 一致；L17 自身放大'
         '（ratio 367）。因果链闭合：语言注入→L17 开关→L24-33 gap 抹平→B 近零。'
         '教训：np.where 条件 (N,1) vs x (N,) 静默广播 (N,N)。'
         'Ledger 106 / hash ' + chk + '。\n')
wl = ''
if os.path.exists(WLOG):
    wl = open(WLOG, encoding='utf-8').read()
if 'Phase 2967' not in wl:
    wl += entry
    open(WLOG, 'w', encoding='utf-8').write(wl)

# ---------------- MEMORY ----------------
mem = open(WMEM, encoding='utf-8').read()
mem = mem.replace(
    '当前 max=**2966**，下一个 **2967**（候选 A B(s) 塌缩载体解剖；B h15 biphasic '
    '峰位表征；C S1 扩容复检；D 卡片扩充）',
    '当前 max=**2967**，下一个 **2968**（候选 A h15 biphasic 峰位表征（逐词峰位 vs '
    's_c、跨方向泛化）；B L34 残差 signature（0.51×gap 头级构成，npz 离线）；'
    'C S1 扩容复检；D 卡片扩充）')
old = '→路由成员：h15 biphasic 峰锁 s_c 非单调成员，B 被语言注入单调抹平(2966)。'
if old in mem:
    mem = mem.replace(
        old,
        old[:-1] + '→塌缩载体：B 抹平由 L24-33 八层 318 头分布式承载，'
                   'L34 显著响应但端点比 0.51 未塌尽(2967)。')
add = ('- np.where 广播陷阱（2967）：np.where(条件, x, 标量) 中条件 (N,1) 与 x (N,) '
       '会静默广播成 (N,N) 矩阵——np.where 的条件与候选必须同形，或改用布尔索引。'
       '科学结论：B 塌缩载体 = L24-33 深层带八层分布式（318/1152 头显著，maxT 族校正，'
       'top12 全完美单调 |rho|=1.0，无单点载体）；L34 显著响应（q 1e-3）但端点比 '
       '0.5148——"响应显著"与"塌缩彻底"是两个量，判据必须分层（显著层/载体层两档）。'
       '同实现同 batch 组成的跨相位 bit 级复制锚（a12-14 vs 2966 全 0.0）第三次验证。\n')
if 'np.where 广播陷阱（2967）' not in mem:
    anchor_line = '- 剂量响应检验形状规范（2966）：'
    i = mem.find(anchor_line)
    if i >= 0:
        mem = mem[:i] + add + mem[i:]
    else:
        mem = mem.rstrip() + '\n' + add
open(WMEM, 'w', encoding='utf-8').write(mem)

print('closeout done: ledger=%s n_meas=%s' % (
    chk, len(led['measurements'])))
